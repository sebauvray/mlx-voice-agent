"""Le cerveau : un LLM joint en HTTP, sur l'infra du LAN.

`mlx_audio.sts.voice_pipeline` ne sait charger qu'un mlx-lm LOCAL, mais
`VoicePipeline(response_engine=...)` est un point d'injection de première
classe : quand on fournit le moteur, le pipeline ne construit jamais le
sien et n'importe même pas mlx-lm.

L'interface à respecter tient en deux méthodes, copiées sur
LocalLLMResponseEngine : `load()` et `generate(transcript, context) -> str`.
C'est tout ce qui nous couple à mlx-audio ; une montée de version ne peut
casser ce fichier que si ces deux signatures bougent.
"""

import json
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Callable, Optional

# On coupe l'historique : un modèle local a une fenêtre de contexte modeste
# et une conversation vocale part vite en cacahuète. 12 messages = 6 tours.
MAX_HISTORY_MESSAGES = 12

# Beaucoup de modèles locaux (Qwen3, R1 et dérivés) émettent leur
# raisonnement dans <think>…</think>. Sans ce filtre, le TTS le LIT À VOIX
# HAUTE — trente secondes de monologue avant la réponse.
_THINK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
# Cas fréquent avec un modèle de raisonnement : max_tokens tombe AVANT la
# balise fermante. Tout ce qui suit un <think> orphelin est du raisonnement
# et part à la poubelle — il ne reste alors rien à dire, ce qui est la
# bonne réponse. (Si ça arrive souvent, le modèle est mal choisi pour du
# vocal : coupe le raisonnement côté endpoint plutôt que de le filtrer.)
_THINK_OPEN = re.compile(r"<(?:think|thinking|reasoning)>.*$", re.DOTALL | re.IGNORECASE)
_MARKDOWN = re.compile(r"[*_`#]+")

# Rôle de secrétaire, pas d'expert. Elle écoute, elle retient, elle fait
# préciser — elle ne conseille pas. C'est délibéré : le raisonnement est
# coupé pour tenir 0,6 s par tour, donc les avis qu'elle donnerait seraient
# superficiels. Mieux vaut qu'elle n'en donne pas et qu'elle capte bien.
DEFAULT_PROMPT = (
    "Tu es la secrétaire de la personne qui te parle. Ton rôle est de "
    "l'écouter développer une idée et de l'aider à la préciser, PAS de "
    "donner ton avis ni de proposer des solutions.\n"
    "Tu accuses réception en une phrase, puis tu poses UNE question courte "
    "qui fait avancer sa réflexion — sur un point resté vague, une "
    "contradiction, ou ce qui manque. Si tout est clair, tu le dis et tu "
    "l'invites à continuer.\n"
    "Deux phrases maximum, jamais trois. Tu parles français, en langage "
    "parlé. Jamais de markdown, jamais d'emoji, jamais de liste : tout ce "
    "que tu écris sera lu à voix haute.\n"
    "Ne récapitule jamais spontanément : la synthèse se fera à la fin, "
    "ailleurs."
)


def speakable(text: str) -> str:
    """Ce qui part au TTS. Le prompt système interdit déjà le markdown ; les
    modèles locaux l'ignorent régulièrement, et un astérisque se prononce
    « astérisque »."""
    text = _THINK.sub("", text)
    text = _THINK_OPEN.sub("", text)
    text = _MARKDOWN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class RemoteLLMResponseEngine:
    """Même interface que LocalLLMResponseEngine : load() + generate()."""

    def __init__(self, url: str, model: str, *, system_prompt: str = DEFAULT_PROMPT,
                 api_key: str = "", timeout: float = 60.0, max_tokens: int = 512,
                 temperature: float = 0.7, thinking: bool = False,
                 on_exchange: Optional[Callable[[str, str, float], None]] = None):
        self.url = url
        self.model = model
        self.system_prompt = system_prompt
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        # Coupé par défaut, et ce n'est pas un choix de confort. Mesuré sur
        # l'endpoint, sur des questions simples ET complexes :
        #
        #   raisonnement activé : 6,54 s, 512 tokens, content VIDE
        #   raisonnement coupé  : 0,55 s,  40 tokens, réponse pertinente
        #
        # Le modèle écrit sa réflexion dans `reasoning_content`, un champ
        # séparé, et épuise systématiquement le budget avant d'atteindre
        # `content`. Le garder, c'est attendre douze fois plus longtemps
        # pour n'entendre rien.
        self.thinking = thinking
        # Le serveur web s'en sert pour pousser la conversation à l'écran.
        self.on_exchange = on_exchange

    def _post(self, payload: dict, timeout: float) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(), headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            # strict=False : l'endpoint renvoie des sauts de ligne BRUTS à
            # l'intérieur des chaînes JSON, ce que le décodeur refuse par
            # défaut. Sans ça, tout appel un peu long meurt sur
            # « Invalid control character » — constaté dès le premier essai.
            return json.loads(resp.read(), strict=False)

    def load(self) -> None:
        """Un aller-retour réel AVANT d'ouvrir le micro. Découvrir que l'URL
        est fausse après avoir parlé dans le vide est le pire des
        diagnostics : le pipeline ne remonte rien, il reste juste muet."""
        try:
            got = self._post(
                {"model": self.model,
                 "messages": [{"role": "user", "content": "ping"}],
                 "max_tokens": 1, "stream": False},
                timeout=min(self.timeout, 20.0),
            )
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:300]
            raise RuntimeError(
                f"Le LLM a répondu HTTP {e.code} sur {self.url}\n  {body}\n"
                f"  Vérifie le nom du modèle (« {self.model} ») et le chemin de l'URL."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"LLM injoignable sur {self.url} : {type(e).__name__}: {e}"
            ) from e
        if "choices" not in got:
            raise RuntimeError(
                f"Réponse inattendue de {self.url} : pas de champ « choices ». "
                f"Cet endpoint est-il bien compatible OpenAI ?\n  Reçu : {str(got)[:300]}"
            )
        print(f"  LLM OK : {self.model} @ {self.url}", file=sys.stderr)

    def generate(self, transcript: str, context: Optional[list] = None) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        if context:
            messages.extend(context[-MAX_HISTORY_MESSAGES:])
        messages.append({"role": "user", "content": transcript})

        started = time.monotonic()
        try:
            payload = {"model": self.model, "messages": messages, "stream": False,
                       "max_tokens": self.max_tokens, "temperature": self.temperature}
            if not self.thinking:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            got = self._post(payload, timeout=self.timeout)
            message = got["choices"][0]["message"]
            raw = message.get("content") or ""
        except Exception as e:
            # On ne fait PAS tomber le pipeline : une requête ratée doit
            # coûter un tour de parole, pas la session entière.
            print(f"\n  ✗ LLM : {type(e).__name__}: {e}", file=sys.stderr)
            return "Désolé, je n'ai pas pu joindre le modèle."

        elapsed = time.monotonic() - started
        answer = speakable(raw)
        if not raw.strip():
            pense = message.get("reasoning_content") or message.get("reasoning") or ""
            if pense:
                # On NOMME la cause : sinon « je n'ai rien reçu » masque un
                # comportement parfaitement identifié.
                print(f"  ! content vide mais {len(pense)} car. de raisonnement — "
                      f"le budget de tokens est parti dedans", file=sys.stderr)
            # Constaté en usage réel : le modèle renvoie parfois une
            # complétion VIDE. Rendre "" fait abandonner le tour au
            # pipeline sans un mot — l'utilisateur attend une réponse qui
            # ne viendra jamais et n'a aucun moyen de comprendre. Mieux
            # vaut dire qu'on n'a rien reçu.
            print(f"  ! le LLM a renvoyé une réponse VIDE "
                  f"(usage: {got.get('usage')})", file=sys.stderr)
            answer = "Je n'ai rien reçu du modèle. Tu peux répéter ?"
        elif not answer:
            # Constaté en usage réel : « thinker » émet parfois un
            # <think> non refermé qui consomme tout le budget de tokens.
            # Le filtre fait son travail, mais il ne reste rien à dire.
            # On le NOMME au lieu de servir une phrase creuse — sinon le
            # symptôme (« Je n'ai rien à dire ») masque sa cause.
            print(f"  ! réponse entièrement filtrée ({len(raw)} car. de "
                  f"raisonnement, max_tokens={self.max_tokens} atteint ?)",
                  file=sys.stderr)
            answer = "Je me suis perdu dans mes pensées. Redis-moi ça ?"
        print(f"\n  ⟳ toi : {transcript}\n  ⟲ lui : {answer}\n  ⏱ LLM : {elapsed:.2f} s",
              file=sys.stderr)
        if self.on_exchange is not None:
            try:
                self.on_exchange(transcript, answer, elapsed)
            except Exception as e:  # l'affichage ne doit jamais tuer la voix
                print(f"  (on_exchange: {type(e).__name__}: {e})", file=sys.stderr)
        return answer


# ══════════════════════════════════════════════════════════════════════
# Le bail sur la carte
# ══════════════════════════════════════════════════════════════════════
# L'infra ne sert pas le modèle à la demande : il faut RÉSERVER le GPU,
# le garder vivant, et le rendre. Trois pièges, chacun payé une fois :
#
#   1. `acquire` répond « go » OU AUTRE CHOSE. « fallback » n'est pas un
#      accord, c'est une contre-proposition — s'en contenter fait échouer
#      trois étapes plus loin, sur une erreur qui ne dit pas sa cause.
#   2. 300 s sans battement et la carte est retirée EN COURS DE ROUTE.
#      Une conversation vocale a des silences bien plus longs que ça.
#      D'où un battement toutes les 60 s, cinq fois la marge.
#   3. Ne pas rendre la carte la laisse bloquée jusqu'à son expiration,
#      pour tout le monde. release() est donc appelé dans un finally ET
#      via atexit — un Ctrl-C ne doit pas pénaliser la machine suivante.

import atexit
import os
import pathlib
import threading

HEARTBEAT_SECONDS = 60
CONF_PATH = pathlib.Path.home() / ".thebureau" / "thebureau.conf"


def load_conf(path: pathlib.Path = CONF_PATH) -> dict:
    """Lit le KEY=value shell de thebureau.conf. On ne `source` pas : ce
    serait exécuter du shell arbitraire pour lire cinq variables."""
    conf = {}
    if not path.exists():
        return conf
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        conf[key] = value.strip().strip('"').strip("'")
    return conf


class GpuLease:
    """Réserve la carte, la maintient, la rend."""

    def __init__(self, base: str, token: str, model: str, session_id: str,
                 on_note=None):
        self.base = base.rstrip("/")
        self.token = token
        self.model = model
        self.session_id = session_id
        self.on_note = on_note or (lambda text: None)
        self.held = False
        self._stop = threading.Event()
        self._beat: Optional[threading.Thread] = None
        atexit.register(self.release)

    def _call(self, path: str, payload: dict, timeout: float) -> dict:
        req = urllib.request.Request(
            f"{self.base}/{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.token}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
        return json.loads(body, strict=False) if body else {}

    def acquire(self, timeout: float = 200.0) -> None:
        """Long-poll : ne rend la main que quand le modèle est en VRAM."""
        started = time.monotonic()
        try:
            got = self._call("acquire",
                             {"session_id": self.session_id, "model": self.model},
                             timeout)
        except Exception as e:
            raise RuntimeError(
                f"Gestionnaire de sessions injoignable ({self.base}) : "
                f"{type(e).__name__}: {e}"
            ) from e

        status = str(got.get("status", "")).lower()
        if status != "go":
            # Refus explicite, y compris « fallback ». On ne devine pas.
            raise RuntimeError(
                f"Carte refusée — statut « {status or '?'} ». "
                f"Seul « go » vaut accord.\n  Réponse : {str(got)[:300]}"
            )

        self.held = True
        waited = time.monotonic() - started
        self.on_note(f"Carte réservée en {waited:.0f} s (modèle « {self.model} »).")
        print(f"  carte réservée en {waited:.0f} s", file=sys.stderr)
        self._stop.clear()
        self._beat = threading.Thread(target=self._heartbeat, daemon=True)
        self._beat.start()

    def _heartbeat(self) -> None:
        # `wait` plutôt que `sleep` : à la libération on veut sortir tout de
        # suite, pas au bout d'une minute.
        while not self._stop.wait(HEARTBEAT_SECONDS):
            try:
                self._call("heartbeat", {"session_id": self.session_id}, 15.0)
            except Exception as e:
                # Un battement raté n'est pas fatal — il en reste quatre
                # avant les 300 s. On le dit sans couper la conversation.
                print(f"  ! battement raté : {type(e).__name__}: {e}", file=sys.stderr)

    def release(self) -> None:
        if not self.held:
            return
        self.held = False
        self._stop.set()
        try:
            self._call("release", {"session_id": self.session_id}, 15.0)
            print("  carte rendue", file=sys.stderr)
        except Exception as e:
            print(f"  ! carte NON rendue : {type(e).__name__}: {e}\n"
                  f"    (elle se libérera d'elle-même à l'expiration)", file=sys.stderr)

