#!/usr/bin/env python3
"""Agent conversationnel vocal, servi en web depuis le Mac mini.

Le micro est là où tu es (téléphone, MacBook), le GPU est ici. On déporte
donc les deux extrémités audio du pipeline vers le navigateur, et rien
d'autre : STT streaming, VAD, détection de fin de tour, TTS et barge-in
restent tels que mlx-audio les implémente.

── Les deux greffes ─────────────────────────────────────────────────
ENTRÉE  `VoicePipeline._listener()` ouvre `sd.InputStream` en dur, sans
        paramètre `device` : sur cette machine il n'y a de toute façon
        aucun micro physique (les cinq entrées sont virtuelles — KVM, NDI,
        Teams, Steam). On surcharge la méthode pour consommer la même
        `input_audio_queue`, alimentée par la WebSocket. Le corps de la
        boucle est identique à l'original.

SORTIE  `AudioOutputStream` est None-safe en `enabled=False` : toutes ses
        méthodes renvoient des valeurs dégénérées. On hérite dans ce
        mode-là et on ne surcharge que `queue_audio` et `flush`, au lieu
        de réimplémenter `echo_correlation`, `playback_state` et le reste.

── Pourquoi l'anti-écho serveur ne manque pas ───────────────────────
Ces valeurs dégénérées désactivent la corrélation d'écho de mlx-audio.
C'est voulu : le serveur ne peut PAS savoir quand le son est réellement
sorti d'un haut-parleur à l'autre bout du LAN. Le navigateur, lui, le
sait — `getUserMedia({echoCancellation: true})` donne l'AEC de WebRTC,
celle de Meet et de Zoom. On annule l'écho là où l'information existe.

── HTTPS obligatoire, et pourquoi ça tombe bien ─────────────────────
`getUserMedia` exige un contexte sécurisé : en http://192.168.x.x le
navigateur refuse le micro, iOS le premier. Un workflow FlowHub qui
déclare `port:` est publié en HTTPS derrière le NDD interne — c'est
comme ça que CloudCLI tourne en PWA sur mobile. D'où l'écoute sur
127.0.0.1 : c'est l'ingress qui expose, pas nous.

── Une seule session à la fois ──────────────────────────────────────
Voxtral pèse 3,15 Go de poids. Une instance de pipeline par onglet
saturerait les 16 Go de la machine. On garde donc UN pipeline, chargé
une fois (~30 s) et gardé chaud entre les connexions, et on refuse la
seconde connexion avec un message clair plutôt que de tomber en OOM.
"""

import argparse
import asyncio
import hmac
import json
import logging
import os
import pathlib
import secrets
import sys
import time
import uuid
from typing import Any, Optional

import ipaddress
import re
import unicodedata

import mlx.core as mx
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse

from mlx_audio.sts.voice_pipeline import (
    AudioOutputStream,
    VoicePipeline,
    VoicePipelineConfig,
)

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from llm_engine import (  # noqa: E402
    DEFAULT_PROMPT, GpuLease, RemoteLLMResponseEngine, load_conf,
)
from notes import Notebook  # noqa: E402
from tts_fr import FrenchKokoroResponder  # noqa: E402

HERE = pathlib.Path(__file__).parent
INDEX = HERE / "static" / "index.html"


class Hub:
    """Le client courant. Tout passe par ici pour n'avoir qu'un seul
    endroit où gérer « personne n'est connecté »."""

    def __init__(self) -> None:
        self.ws: Optional[WebSocket] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.outbox: asyncio.Queue = asyncio.Queue(maxsize=256)

    def attach(self, ws: WebSocket, loop) -> None:
        self.ws, self.loop = ws, loop

    def detach(self) -> None:
        self.ws = None
        while not self.outbox.empty():          # pas de son d'une session
            self.outbox.get_nowait()            # dans la suivante

    def push(self, item: Any) -> None:
        """Appelable depuis n'importe quel thread : MLX tourne dans un
        pool, l'event loop non. `call_soon_threadsafe` est le seul pont
        correct entre les deux."""
        if self.ws is None or self.loop is None:
            return
        def _put():
            try:
                self.outbox.put_nowait(item)
            except asyncio.QueueFull:
                pass                            # on jette le son, pas la session
        try:
            self.loop.call_soon_threadsafe(_put)
        except RuntimeError:
            pass                                # boucle déjà fermée

    def event(self, kind: str, **fields: Any) -> None:
        self.push({"type": kind, **fields})


HUB = Hub()


class WebAudioOutput(AudioOutputStream):
    """La sortie TTS part sur la WebSocket au lieu du haut-parleur local."""

    def __init__(self, *, sample_rate: int):
        super().__init__(sample_rate=sample_rate, enabled=False)

    def queue_audio(self, audio) -> dict:
        HUB.push(_to_pcm16(audio))
        if "first_audio" not in CHRONO.marks:
            # Le premier bloc SORTANT, pas le début de la synthèse : c'est
            # l'instant où le son part vraiment vers le navigateur.
            CHRONO.mark("first_audio")
            CHRONO.report()
        return super().queue_audio(audio)

    def flush(self) -> dict:
        # Barge-in : le navigateur a peut-être déjà plusieurs secondes en
        # tampon. Sans cet ordre explicite il continuerait de parler après
        # qu'on lui a coupé la parole — le défaut le plus visible qui soit.
        HUB.event("flush")
        return super().flush()


def _to_pcm16(audio) -> bytes:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class Chrono:
    """La décomposition réelle de l'attente, tour par tour.

    Écrite parce que deux optimisations de suite ont visé à côté : la
    finalisation STT mesurée sur banc de test était fausse (l'audio y
    arrivait moins vite que le temps réel), et le streaming du LLM ne
    rapportait que 0,17 s (le coût est AVANT le premier token, pas dans
    la génération). On ne devine plus, on lit le journal."""

    def __init__(self) -> None:
        self.marks: dict = {}

    def mark(self, name: str) -> None:
        self.marks[name] = time.monotonic()

    def reset(self) -> None:
        self.marks.clear()

    def report(self) -> None:
        m = self.marks
        t0 = m.get("turn_end")
        if t0 is None:
            return
        def gap(a, b):
            if a not in m or b not in m:
                return "   —   "
            return f"{m[b] - m[a]:6.2f}s"
        total = (m.get("first_audio") or m.get("tts_started") or t0) - t0
        go = 1024 ** 3
        print(f"  ⧉ MLX : actif {mx.get_active_memory()/go:.1f} Go"
              f" | cache {mx.get_cache_memory()/go:.1f} Go"
              f" | pic {mx.get_peak_memory()/go:.1f} Go",
              file=sys.stderr)
        print(f"  ⏲ depuis la fin de parole :"
              f" STT {gap('turn_end', 'transcript')}"
              f" | LLM {gap('transcript', 'llm_done')}"
              f" | TTS {gap('llm_done', 'first_audio')}"
              f" | TOTAL {total:6.2f}s"
              f"  (+{VAD_SILENCE_MS} ms de silence VAD en amont)",
              file=sys.stderr)


def _same_utterance(a: str, b: str) -> bool:
    """Deux transcriptions disent-elles la même chose ?

    On ignore casse, accents, ponctuation et espaces : la finalisation
    ajoute presque toujours un point d'interrogation ou une virgule, ce
    qui ne change pas le sens. En revanche un mot manquant, si — « peux-tu
    éteindre » et « peux-tu éteindre la lumière » n'appellent pas la même
    réponse. La comparaison reste donc stricte sur les mots."""
    def norm(t: str) -> str:
        t = unicodedata.normalize("NFD", t.lower())
        t = "".join(c for c in t if unicodedata.category(c) != "Mn")
        return re.sub(r"[^a-z0-9]+", " ", t).strip()
    return norm(a) == norm(b)


CHRONO = Chrono()
VAD_SILENCE_MS = 600


class WebVoicePipeline(VoicePipeline):
    async def _listener(self) -> None:
        """Même boucle que l'originale, sans le sd.InputStream : les trames
        arrivent de la WebSocket."""
        HUB.event("state", value="listening")
        while True:
            samples = await self.input_audio_queue.get()
            await self._process_input_audio(samples)
            self.input_audio_queue.task_done()

    def _log_event(self, event: str, **fields: Any) -> None:
        # ATTENTION : l'implémentation d'origine sort immédiatement quand
        # verbose est faux. Notre travail doit donc précéder l'appel à
        # super(), sinon l'UI ne reçoit rien hors mode verbeux.
        # Jalons du chronomètre. `endpoint_candidate` avec complete=true est
        # l'instant où le tour est déclaré fini : c'est de là que compte
        # l'attente ressentie, aux ~600 ms de silence VAD près qui la
        # précèdent et qu'on ne peut pas supprimer sans couper la parole.
        if event == "endpoint_candidate":
            # Toujours journalisé, pas seulement en verbeux : c'est LA
            # décision qui coupe la parole. Sans la probabilité sous les
            # yeux, régler le seuil se fait au jugé.
            print(f"  ⟂ fin de tour ? {str(fields.get('complete')):5s}"
                  f" p={float(fields.get('probability', 0)):.2f}"
                  f" seuil={TURN_THRESHOLD:.2f}"
                  f" silence={float(fields.get('silence_ms', 0)):.0f}ms"
                  f" audio={float(fields.get('audio_ms', 0))/1000:.1f}s",
                  file=sys.stderr)

        if event == "endpoint_candidate":
            # POURQUOI le tour se termine. Deux chemins mènent ici : soit
            # smart-turn juge la phrase finie, soit le silence a dépassé
            # turn_max_incomplete_silence_ms et on commite malgré lui.
            # Sans cette trace, régler les seuils revient à tourner des
            # boutons en aveugle.
            # « évaluation » et non « fin de tour » : ce journal est émis à
            # CHAQUE examen, y compris ceux qui concluent « pas fini » et
            # laissent le tour ouvert. Le libellé précédent laissait croire
            # à une coupure là où il n'y en avait pas.
            done = str(fields.get("complete", "")).lower() == "true"
            print(f"  ⊣ évaluation : {'FIN' if done else 'pas fini'}"
                  f" (p={fields.get('probability', 0):.2f},"
                  f" seuil {TURN_THRESHOLD:.2f},"
                  f" silence {fields.get('silence_ms', 0):.0f} ms)",
                  file=sys.stderr)

        if event == "turn_finalization_started":
            CHRONO.reset(); CHRONO.mark("turn_end")
        elif event == "turn_finalized":
            CHRONO.mark("transcript")
        elif event == "response_ready":
            CHRONO.mark("llm_done")
        elif event == "tts_started":
            CHRONO.mark("tts_started")

        if event == "partial_transcript_delta":
            HUB.event("partial", text=fields.get("text", ""))
        elif event == "tts_started":
            HUB.event("state", value="speaking")
        elif event == "barge_in_confirmed":
            HUB.event("state", value="listening")
        elif event == "turn_dropped":
            HUB.event("state", value="listening")
        elif event == "tts_error":
            HUB.event("error", text=str(fields.get("error", "TTS en échec")))
        if self.config.verbose:
            # Canal de diagnostic : sans lui, un tour qui ne se termine pas
            # ne laisse aucune trace exploitable côté client.
            HUB.event("dbg", event=event,
                      fields={k: str(v)[:120] for k, v in fields.items()})
        super()._log_event(event, **fields)

    # ── Anticipation ────────────────────────────────────────────────
    # La finalisation Voxtral coûte 1 à 2 s, pendant lesquelles on a DÉJÀ
    # une transcription partielle quasi complète — il ne lui manque en
    # général que le dernier mot et la ponctuation. On lance donc l'appel
    # au LLM dessus, en parallèle du travail de finalisation.
    #
    # Le pari est SANS RISQUE de latence : si le texte final diffère, on
    # jette et on refait l'appel au moment exact où on l'aurait fait de
    # toute façon. On ne perd que du calcul côté infra, jamais du temps.
    #
    # On ne parle JAMAIS d'une réponse spéculative non validée : la
    # comparaison a lieu avant de synthétiser quoi que ce soit.

    async def _finalize_turn(self) -> None:
        partial = (getattr(self.transcriber, "text", "") or "").strip()
        self._spec = None
        if SPECULATE and len(partial) >= 12 and self.response_engine is not None:
            conversation = list(self._conversation)
            self._spec = (
                partial,
                asyncio.create_task(asyncio.to_thread(
                    self.response_engine.generate, partial, conversation)),
            )
        await super()._finalize_turn()

    async def _respond_to_transcript(self, transcript: str) -> None:
        """Copie fidèle de l'original, à UN détail près : l'appel au LLM
        part dans un thread quelconque au lieu du thread MLX.

        L'original fait `self.mlx.run(...)`. Or ce thread est unique et
        porte AUSSI le VAD et le STT — les streams MLX y sont liés. Une
        requête HTTP de 1,4 s y gèle donc la détection de parole, et le
        barge-in ne répond plus pendant que l'assistant réfléchit. Or
        `generate()` ne touche pas à MLX : il n'a rien à faire là."""
        spec, self._spec = getattr(self, "_spec", None), None
        response_text = None
        if spec is not None:
            partial, task = spec
            if _same_utterance(partial, transcript):
                response_text = await task
                print(f"  ⚡ anticipation retenue ({len(partial)} car.)", file=sys.stderr)
            else:
                task.cancel()
                print(f"  ⚡ anticipation rejetée : « {partial} » ≠ « {transcript} »",
                      file=sys.stderr)
        if response_text is None:
            response_text = await asyncio.to_thread(
                self.response_engine.generate, transcript, self._conversation
            )
        response_text = response_text.strip()
        if not response_text:
            return
        self._current_response_text = response_text
        self._log_event("response_ready", text=response_text, chars=len(response_text))
        self._conversation.append({"role": "user", "content": transcript})
        self._conversation.append({"role": "assistant", "content": response_text})
        self._current_tts_cancel = asyncio.Event()
        self._current_tts_task = asyncio.create_task(
            self._speak_response(response_text, self._current_tts_cancel)
        )

    # ── Le tour qui ne se terminait jamais ──────────────────────────
    # `candidate_ended` est un FRONT : le VAD le lève une seule fois, au
    # passage parole→silence, puis bascule `in_speech` à False et ne peut
    # plus le relever tant qu'on n'a pas reparlé.
    #
    # `_maybe_finalize_turn` n'est donc appelé qu'UNE fois par tour — et à
    # cet instant `silence_ms` vaut 0, puisqu'il vient d'être initialisé.
    # La condition `silence_ms >= turn_max_incomplete_silence_ms` ne peut
    # jamais être vraie : ce paramètre est inopérant en amont.
    #
    # Résultat : seul smart-turn au-dessus du seuil termine un tour. Si tu
    # finis une phrase qu'il juge inachevée, plus rien ne se passe — il
    # faut reparler pour provoquer un nouveau front. C'est ce qui a rendu
    # l'agent muet quand j'ai monté le seuil de 0,7 à 0,9.
    #
    # Ce chien de garde redonne son sens au paramètre : tant que le tour
    # reste ouvert, on redemande son avis à smart-turn à intervalle
    # régulier, avec un silence qui s'allonge. Il finit donc soit par
    # dépasser le seuil, soit par atteindre le délai.

    _WATCHDOG_INTERVAL = 0.4

    async def _maybe_finalize_turn(self, now: float) -> None:
        await super()._maybe_finalize_turn(now)
        if TURN_WATCHDOG:
            self._arm_turn_watchdog()

    def _arm_turn_watchdog(self) -> None:
        en_cours = getattr(self, "_turn_watchdog", None)
        if en_cours is not None and not en_cours.done():
            return
        if self.transcriber is None or self.transcriber.session is None:
            return  # le tour vient d'être finalisé, rien à surveiller
        self._turn_watchdog = asyncio.create_task(self._watch_turn())

    async def _watch_turn(self) -> None:
        limite = (self.config.turn_max_incomplete_silence_ms / 1000.0) + 1.0
        écoulé = 0.0
        try:
            while écoulé < limite:
                await asyncio.sleep(self._WATCHDOG_INTERVAL)
                écoulé += self._WATCHDOG_INTERVAL
                # Reparlé, ou déjà finalisé : plus rien à surveiller.
                if self.transcriber is None or self.transcriber.session is None:
                    return
                if getattr(self.speech_gate, "in_speech", False):
                    return
                if not self._turn_audio:
                    return
                await super()._maybe_finalize_turn(time.monotonic())
        except asyncio.CancelledError:
            raise

    def feed(self, pcm16: bytes) -> None:
        """Trames micro du navigateur : int16 mono 16 kHz."""
        samples = self._normalize_input(np.frombuffer(pcm16, dtype=np.int16))
        try:
            self.input_audio_queue.put_nowait(samples)
        except asyncio.QueueFull:
            pass                                # on préfère un trou à un retard


# Désactivée par défaut : mesurée à 0 succès sur 3, parce que la
# transcription partielle s'arrête 2 à 4 mots avant la fin — pas
# seulement avant la ponctuation. Et chaque tentative coûte un appel LLM
# de plus à l'infra. À rallumer si la mesure en usage réel dit autre
# chose : le journal indique « anticipation retenue/rejetée » à chaque
# tour.
SPECULATE = False
# 0,5 (défaut de mlx-audio) coupe la parole à qui marque une pause pour
# réfléchir : constaté en usage réel, phrases tronquées en plein milieu.
# À 0,7 smart-turn doit être nettement plus sûr pour couper ; dans le
# doute il attend `turn_max_incomplete_silence_ms`, ce qui coûte au pire
# une seconde et demie sur une fin ambiguë — bien moins gênant qu'une
# phrase amputée.
TURN_THRESHOLD = 0.7
# Plages d'où l'on accepte une requête sans jeton, parce qu'elle ne peut
# venir que du reverse proxy. Mesuré sur cette machine : Caddy se présente
# en 192.168.97.1, le pont d'OrbStack. Surchargeable, car un renumérotage
# du pont referme l'accès sans prévenir — le journal l'annonce au démarrage.
# Éteint par défaut. Le chien de garde fait bien ce qu'on attend de lui —
# il rend son effet à turn_max_incomplete_silence_ms, vérifié : les
# réévaluations s'enchaînent (0, 413, 1703, 3316 ms) et la finalisation
# démarre au franchissement du seuil. MAIS `transcriber.finish()` ne rend
# alors jamais la main : ce chemin n'était jamais emprunté en amont, et il
# s'y bloque. Tant que ce n'est pas élucidé, on ne l'impose pas.
TURN_WATCHDOG = False
INGRESS_NETS: list = []
NOTEBOOK = Notebook()
TTS_LANG = "f"
PIPELINE: Optional[WebVoicePipeline] = None
PIPELINE_TASK: Optional[asyncio.Task] = None


async def ensure_pipeline(cfg: VoicePipelineConfig, engine) -> WebVoicePipeline:
    """Charge les modèles une seule fois et garde le pipeline chaud."""
    global PIPELINE, PIPELINE_TASK
    if PIPELINE is not None:
        return PIPELINE

    pipeline = WebVoicePipeline(cfg, response_engine=engine)
    # init_models() est idempotent (tout est « if X is None »), donc on peut
    # l'appeler ici pour connaître le vrai taux d'échantillonnage du TTS
    # avant d'injecter la sortie. Le start() suivant ne le refera pas.
    await pipeline.init_models()

    # On garde le modèle que init_models vient de charger SUR LE THREAD MLX
    # (les streams MLX sont liés à leur thread) et on ne remplace que
    # l'enveloppe, qui elle ne calcule rien. Voir tts_fr.py pour le
    # pourquoi : sans lang_code, Kokoro lit le français à l'anglaise.
    # Le même Kokoro sert la conversation ET /v1/audio/speech. On le
    # dépose dans le cache du serveur mlx-audio pour qu'il le réutilise au
    # lieu d'en charger un second : c'est tout l'intérêt d'un process
    # unique, et ça vaut ~370 Mo.
    try:
        from mlx_audio.server import model_provider
        model_provider.models.setdefault(cfg.tts_model, pipeline.tts_responder.model)
    except Exception as e:
        print(f"  ! partage du TTS impossible : {type(e).__name__}: {e}",
              file=sys.stderr)

    pipeline.tts_responder = FrenchKokoroResponder(
        pipeline.tts_responder.model,
        voice=cfg.tts_voice,
        lang_code=TTS_LANG,
        streaming_interval=float(cfg.tts_streaming_interval),
    )
    await pipeline.mlx.run(pipeline.tts_responder.warmup)

    pipeline.audio_output = WebAudioOutput(sample_rate=pipeline.tts_responder.sample_rate)

    PIPELINE = pipeline
    PIPELINE_TASK = asyncio.create_task(pipeline.start())
    return pipeline


def build_app(cfg: VoicePipelineConfig, engine, make_lease, token: str) -> FastAPI:
    app = FastAPI(title="voice-agent POC")

    # `tb expose` publie sur Internet, pas sur le LAN : les sous-domaines
    # de auvraysebastien.fr passent par un tunnel Cloudflare. Un agent
    # vocal branché sur le LLM privé n'a rien à faire en accès libre, et
    # le pipeline n'a aucune authentification à lui. D'où ce jeton — pas
    # de la sécurité sérieuse, mais de quoi ne pas laisser la porte ouverte
    # le temps d'une POC. `compare_digest` plutôt que `==` : la comparaison
    # ne doit pas fuiter la longueur du préfixe correct.
    def allowed(supplied: Optional[str]) -> bool:
        return hmac.compare_digest(supplied or "", token)

    def par_ingress(request: Request) -> bool:
        """La requête est-elle arrivée par Caddy, donc par le hub ?

        Le manifeste FlowHub ne permet pas d'accrocher `?k=…` à l'URL d'une
        carte : cliquer dessus arriverait donc toujours sur un 403. On
        dispense du jeton ce qui vient du pont des containers — seul Caddy
        y a une patte, et le hub sera lui-même derrière son authentification.

        L'accès DIRECT au port depuis le LAN garde son jeton : c'est le
        chemin qu'on ne contrôle pas.

        En cas de doute (adresse absente, plage renumérotée), on retombe sur
        l'exigence du jeton : ce garde-fou échoue en se fermant, jamais en
        s'ouvrant."""
        if not INGRESS_NETS or request.client is None:
            return False
        try:
            ip = ipaddress.ip_address(request.client.host)
        except ValueError:
            return False
        return any(ip in net for net in INGRESS_NETS)

    @app.get("/whoami")
    async def whoami(request: Request):
        # Diagnostic temporaire : quelle adresse source le serveur voit-il
        # selon qu'on passe par l'ingress ou en direct sur le LAN ?
        return PlainTextResponse(
            f"client={request.client.host if request.client else '?'} "
            f"xff={request.headers.get('x-forwarded-for', '-')}")

    @app.get("/health")
    async def health():
        # Volontairement SANS jeton : le status_check du workflow devrait
        # sinon trimballer le secret dans sa ligne de commande, visible de
        # tout `ps`. Cette route ne révèle rien que le port n'ait déjà dit.
        return PlainTextResponse("ok")

    # Le jeton se présente en ?k=… la première fois, puis vit dans un
    # cookie. Sans ça, « Ajouter à l'écran d'accueil » figerait le secret
    # dans le raccourci et il faudrait le retaper à chaque visite.
    COOKIE = "voice_k"

    @app.get("/")
    async def index(request: Request, k: str = ""):
        supplied = k or request.cookies.get(COOKIE, "")
        if not allowed(supplied) and not par_ingress(request):
            return PlainTextResponse("Jeton absent ou invalide.", status_code=403)
        resp = FileResponse(INDEX)
        # httponly : le JavaScript de la page n'a aucune raison de lire le
        # jeton. samesite=lax suffit — la WebSocket part de cette origine.
        resp.set_cookie(COOKIE, token, httponly=True, samesite="lax",
                        max_age=30 * 24 * 3600)
        return resp

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        depuis_ingress = (
            ws.client is not None
            and INGRESS_NETS
            and any(ipaddress.ip_address(ws.client.host) in n for n in INGRESS_NETS)
        )
        if not allowed(ws.query_params.get("k") or ws.cookies.get(COOKIE)) \
                and not depuis_ingress:
            await ws.close(code=4403)
            return
        await ws.accept()
        if HUB.ws is not None:
            await ws.send_json({
                "type": "busy",
                "text": "Une session vocale est déjà ouverte ailleurs. "
                        "Ferme-la avant d'en ouvrir une autre : les modèles "
                        "tiennent 4 Go et la machine en a 16.",
            })
            await ws.close()
            return

        loop = asyncio.get_running_loop()
        HUB.attach(ws, loop)
        sender = asyncio.create_task(_sender(ws))
        lease = None
        try:
            NOTEBOOK.open()
            await ws.send_json({"type": "state", "value": "loading"})
            pipeline = await ensure_pipeline(cfg, engine)

            if make_lease is not None:
                lease = make_lease()
                await ws.send_json({"type": "note",
                                    "text": "Réservation de la carte…"})
                # acquire long-poll jusqu'à 200 s : dans un thread, sinon il
                # gèlerait la boucle et le navigateur croirait à une panne.
                await asyncio.to_thread(lease.acquire)

            await ws.send_json({"type": "ready",
                                "sample_rate": pipeline.audio_output.sample_rate,
                                "input_sample_rate": cfg.input_sample_rate})
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if (data := msg.get("bytes")) is not None:
                    pipeline.feed(data)
                elif (text := msg.get("text")) is not None:
                    # Le navigateur annonce le taux que son AudioContext lui
                    # a réellement accordé. Safari ignore couramment la
                    # demande de 16 kHz : savoir qu'on rééchantillonne depuis
                    # 48 kHz explique une transcription douteuse.
                    try:
                        info = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if info.get("type") == "client":
                        rate = info.get("rate")
                        note = ("natif" if rate == cfg.input_sample_rate
                                else f"rééchantillonné vers {cfg.input_sample_rate}")
                        print(f"  micro navigateur : {rate} Hz ({note})",
                              file=sys.stderr)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"  ✗ session : {type(e).__name__}: {e}", file=sys.stderr)
            # L'erreur doit arriver à l'écran : sinon l'utilisateur voit un
            # « chargement… » qui ne finit jamais et ne sait pas pourquoi.
            try:
                await ws.send_json({"type": "fatal", "text": str(e)})
            except Exception:
                pass
        finally:
            sender.cancel()
            # La carte se rend AVANT tout le reste, et quoi qu'il arrive :
            # la garder bloquée pénalise la machine suivante.
            if lease is not None:
                await asyncio.to_thread(lease.release)
            chemin = NOTEBOOK.close()
            if chemin is not None:
                HUB.event("note", text=f"Note enregistrée : {chemin.name}")
            HUB.detach()
            # Le pipeline reste chaud, mais la conversation repart à zéro :
            # reprendre le fil d'une session close serait déroutant.
            if PIPELINE is not None:
                PIPELINE._conversation.clear()
            print("  session fermée", file=sys.stderr)

    return app


async def _sender(ws: WebSocket) -> None:
    """Un seul writer sur la socket : Starlette n'aime pas les envois
    concurrents, et le TTS comme les événements poussent en parallèle."""
    while True:
        item = await HUB.outbox.get()
        if isinstance(item, bytes):
            await ws.send_bytes(item)
        else:
            await ws.send_json(item)


def main() -> int:
    # Les valeurs par défaut viennent de ~/.thebureau/thebureau.conf : la
    # conf de l'infra fait autorité, la ligne de commande ne sert qu'à en
    # dévier ponctuellement.
    # Préfixe VOICE_AGENT_ et non VOICE_ : cette machine définit déjà
    # VOICE_TTS_VOICE, VOICE_TTS_MODEL, VOICE_STT_MODEL et
    # VOICE_API_BASE_URL pour la dictée de CloudCLI. Partager le préfixe
    # faisait repasser la voix en af_heart — anglophone — sans un mot.
    conf = load_conf()
    api_base = conf.get("LOCAL_API_BASE", "").rstrip("/")

    p = argparse.ArgumentParser(description="POC agent vocal web")
    p.add_argument("--llm-url",
                   default=os.environ.get("VOICE_AGENT_LLM_URL")
                   or (f"{api_base}/v1/chat/completions" if api_base else ""),
                   help="URL complète, compatible OpenAI (…/v1/chat/completions)")
    p.add_argument("--llm-model", default=os.environ.get("VOICE_AGENT_LLM_MODEL")
                   or conf.get("LOCAL_MODEL", ""))
    p.add_argument("--llm-key", default=os.environ.get("VOICE_AGENT_LLM_KEY")
                   or conf.get("LOCAL_API_KEY", ""))
    p.add_argument("--session-manager", default=conf.get("SESSION_MANAGER_BASE", ""),
                   help="Gestionnaire du bail sur la carte.")
    p.add_argument("--session-token", default=conf.get("SESSION_MANAGER_TOKEN", ""))
    p.add_argument("--no-lease", action="store_true",
                   help="Ne pas réserver la carte (endpoint toujours disponible).")
    p.add_argument("--llm-timeout", type=float, default=60.0)
    p.add_argument("--max-tokens", type=int, default=512,
                   help="En vocal, une réponse de 800 mots est insupportable. "
                        "512 et non 300 : un modèle qui raisonne dépense son "
                        "budget en <think> avant de répondre, et une réponse "
                        "tronquée dans le raisonnement ne laisse rien à dire.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--thinking", action="store_true",
                   default=os.environ.get("VOICE_AGENT_THINKING") == "1",
                   help="Laisser le modèle raisonner avant de répondre. "
                        "Coupé par défaut : mesuré à 6,54 s pour une réponse "
                        "VIDE, contre 0,55 s et une réponse pertinente sans. "
                        "Le raisonnement part dans reasoning_content et épuise "
                        "max_tokens avant d'atteindre content.")
    p.add_argument("--system-prompt", default=DEFAULT_PROMPT)
    p.add_argument("--latency-profile",
                   default=os.environ.get("VOICE_AGENT_LATENCY_PROFILE") or "fast",
                   choices=["fast", "balanced", "accurate"])
    p.add_argument("--tts-model", default=os.environ.get("VOICE_AGENT_TTS_MODEL")
                   or "mlx-community/Kokoro-82M-bf16",
                   help="Kokoro par défaut : c'est le seul des deux moteurs "
                        "livrés avec une voix française.")
    p.add_argument("--voice", default=os.environ.get("VOICE_AGENT_TTS_VOICE") or "ff_siwis")
    p.add_argument("--tts-lang", default=os.environ.get("VOICE_AGENT_TTS_LANG") or "f",
                   help="Code langue Kokoro. 'f' = français. Sans lui, "
                        "Kokoro phonémise à l'anglaise quelle que soit la voix.")
    p.add_argument("--turn-threshold", type=float,
                   default=float(os.environ.get("VOICE_AGENT_TURN_THRESHOLD") or 0.7),
                   help="Confiance exigée de smart-turn pour déclarer ta "
                        "phrase finie. 0,9 et non le 0,5 d'origine : relevé en "
                        "usage réel, smart-turn a coupé une phrase finissant "
                        "sur une virgule avec p=0,83, alors que les vraies fins "
                        "de phrase sortaient à 0,93 et 0,99. Le prix est une "
                        "attente pouvant aller à turn-incomplete-ms quand il "
                        "hésite autour de 0,8 en ayant raison.")
    p.add_argument("--turn-incomplete-ms", type=int,
                   default=int(os.environ.get("VOICE_AGENT_TURN_INCOMPLETE_MS") or 2800),
                   help="Silence toléré quand smart-turn juge la phrase "
                        "INACHEVÉE, avant de commiter quand même. Le défaut "
                        "amont (1600) coupe une pause de réflexion en pleine "
                        "phrase. Ne coûte RIEN sur un tour normal : ce délai "
                        "ne s'applique que si smart-turn dit « pas fini ».")
    p.add_argument("--vad-end-silence-ms", type=int,
                   default=int(os.environ.get("VOICE_AGENT_VAD_SILENCE_MS") or 600),
                   help="Silence avant que smart-turn soit consulté. Se paie "
                        "intégralement sur l'attente ressentie ; le baisser "
                        "rend l'agent plus vif mais plus coupeur de parole.")
    p.add_argument("--host", default="127.0.0.1",
                   help="127.0.0.1 par défaut : c'est l'ingress FlowHub qui "
                        "publie en HTTPS, ce serveur n'a aucune authentification.")
    p.add_argument("--port", type=int, default=8140)
    p.add_argument("--api-port", type=int,
                   default=int(os.environ.get("VOICE_AGENT_API_PORT") or 8131),
                   help="Port de l'API mlx-audio (dictée, synthèse) servie "
                        "par le MÊME process. Sur 127.0.0.1 uniquement : "
                        "elle n'a aucune authentification et ne doit pas "
                        "être publiée par l'ingress.")
    p.add_argument("--no-api", action="store_true",
                   help="Ne pas servir l'API mlx-audio (voix seule).")
    p.add_argument("--token", default=os.environ.get("VOICE_AGENT_TOKEN", ""),
                   help="Jeton d'accès. Généré si absent.")
    p.add_argument("--mlx-cache-gb", type=float,
                   default=float(os.environ.get("VOICE_AGENT_MLX_CACHE_GB") or 2),
                   help="Plafond du cache mémoire de MLX, en Go. 0 = illimité.\n"
                        "Défaut 2 Go, et ce n'est pas arbitraire : relevé en "
                        "usage réel, le cache était monté à 8 Go pour 3,3 Go "
                        "de modèles réellement actifs. Sur une machine de 16 Go "
                        "qui héberge aussi une VM de containers, cette "
                        "thésaurisation pousse le système au swap et "
                        "l'inférence s'effondre. MLX garde ses tampons pour "
                        "éviter de réallouer ; on lui laisse de quoi le faire, "
                        "pas de quoi manger la machine.")
    p.add_argument("--speculate", action="store_true",
                   default=os.environ.get("VOICE_AGENT_SPECULATE") == "1",
                   help="Lance le LLM sur la transcription PARTIELLE, en "
                        "parallèle de la finalisation. Sans risque de latence "
                        "(on rejette et on refait), mais double la charge LLM. "
                        "Mesuré sans effet ici : le partiel manque des mots.")
    p.add_argument("--turn-watchdog", action="store_true",
                   default=os.environ.get("VOICE_AGENT_TURN_WATCHDOG") == "1",
                   help="Réévaluer la fin de tour pendant le silence, pour "
                        "rendre son effet à --turn-incomplete-ms. Fonctionne, "
                        "mais la finalisation qui en découle se bloque dans "
                        "Voxtral — chemin jamais emprunté en amont. Éteint "
                        "tant que ce n'est pas résolu.")
    p.add_argument("--ingress-cidr",
                   default=os.environ.get("VOICE_AGENT_INGRESS_CIDR")
                   or "192.168.97.0/24,192.168.215.0/24,172.17.0.0/16",
                   help="Plages considérées comme « venant du reverse proxy », "
                        "dispensées de jeton. Le manifeste FlowHub ne peut pas "
                        "mettre le jeton dans l'URL d'une carte ; sans ça, "
                        "cliquer dessus donne un 403.")
    p.add_argument("--notes-dir",
                   default=os.environ.get("VOICE_AGENT_NOTES_DIR"),
                   help="Où déposer les notes vocales (défaut : "
                        "~/.mlx-audio/notes). Une note par session, écrite "
                        "à chaque tour et non à la fin : une déconnexion ne "
                        "doit pas coûter la conversation.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.verbose:
        # voice_pipeline journalise ses événements en INFO ; sans ça la
        # racine reste à WARNING et --verbose ne montre rien.
        logging.getLogger().setLevel(logging.INFO)

    if not args.llm_url or not args.llm_model:
        p.error("--llm-url et --llm-model sont requis "
                "(ou VOICE_LLM_URL / VOICE_LLM_MODEL dans l'environnement).")
    if not INDEX.exists():
        p.error(f"Interface introuvable : {INDEX}")

    make_lease = None
    if not args.no_lease:
        if not args.session_manager or not args.session_token:
            p.error("Pas de gestionnaire de sessions dans la conf. "
                    "Passe --session-manager/--session-token, ou --no-lease "
                    "si ton endpoint ne demande pas de réservation.")
        # Un bail par session vocale, avec son propre identifiant : deux
        # sessions ne doivent jamais partager un session_id, sinon le
        # release de l'une retire la carte à l'autre.
        def make_lease():
            return GpuLease(
                args.session_manager, args.session_token, args.llm_model,
                f"voice-{uuid.uuid4().hex[:10]}",
                on_note=lambda text: HUB.event("note", text=text),
            )

    engine = RemoteLLMResponseEngine(
        args.llm_url, args.llm_model,
        system_prompt=args.system_prompt, api_key=args.llm_key,
        timeout=args.llm_timeout, max_tokens=args.max_tokens,
        temperature=args.temperature, thinking=args.thinking,
        on_exchange=lambda user, bot, secs: (
            NOTEBOOK.add(user, bot),
            HUB.event("exchange", user=user, assistant=bot,
                      llm_ms=round(secs * 1000)),
        ),
    )
    global TTS_LANG
    TTS_LANG = args.tts_lang
    global VAD_SILENCE_MS, SPECULATE, TURN_THRESHOLD, NOTEBOOK, INGRESS_NETS
    global TURN_WATCHDOG
    VAD_SILENCE_MS = args.vad_end_silence_ms
    SPECULATE = args.speculate
    TURN_THRESHOLD = args.turn_threshold
    TURN_WATCHDOG = args.turn_watchdog
    if args.notes_dir:
        NOTEBOOK = Notebook(args.notes_dir)
    for cidr in (args.ingress_cidr or "").split(","):
        cidr = cidr.strip()
        if not cidr:
            continue
        try:
            INGRESS_NETS.append(ipaddress.ip_network(cidr))
        except ValueError:
            print(f"  ! plage ingress ignorée : « {cidr} »", file=sys.stderr)
    cfg = VoicePipelineConfig(
        latency_profile=args.latency_profile,
        tts_model=args.tts_model,
        tts_voice=args.voice,
        system_prompt=args.system_prompt,
        vad_end_silence_ms=args.vad_end_silence_ms,
        turn_max_incomplete_silence_ms=args.turn_incomplete_ms,
        turn_threshold=args.turn_threshold,
        verbose=args.verbose,
        play_audio=False,           # le son sort par la WebSocket
        # 4 s de tampon par défaut. L'appel LLM tourne sur le thread MLX,
        # celui qui fait aussi le VAD : pendant ses ~1,5 s le traitement
        # audio est gelé et les trames s'empilent. 256 trames = 8 s, de quoi
        # encaisser un coup de mou de l'infra sans perdre de son.
        queue_size=256,
    )

    if args.mlx_cache_gb > 0:
        mx.set_cache_limit(int(args.mlx_cache_gb * 1024 ** 3))
        print(f"  cache MLX plafonné à {args.mlx_cache_gb:.1f} Go", file=sys.stderr)

    token = args.token or secrets.token_urlsafe(16)
    print(f"\n  Voix      : {args.voice} ({args.tts_model}, langue « {args.tts_lang} »)\n"
          f"  LLM       : {args.llm_model} @ {args.llm_url}"
          f" (raisonnement {'activé' if args.thinking else 'coupé'})\n"
          f"  Carte     : {'réservée par session' if make_lease else 'pas de réservation'}\n"
          f"  Fin de tour : smart-turn p≥{args.turn_threshold}"
          f"{f', sinon {args.turn_incomplete_ms} ms' if args.turn_watchdog else ' (pas de repli sur délai)'}\n"
          f"  API mlx-audio : "
          f"{'désactivée' if args.no_api else f'127.0.0.1:{args.api_port} (locale)'}\n"
          f"  Sans jeton depuis : {', '.join(str(n) for n in INGRESS_NETS) or 'nulle part'}\n"
          f"  Notes     : {NOTEBOOK.dir}\n"
          f"  Interface : http://{args.host}:{args.port}/?k={token}\n"
          f"  (le micro exige HTTPS : passe par une URL https, pas par l'IP du LAN)\n",
          file=sys.stderr)
    asyncio.run(_serve(build_app(cfg, engine, make_lease, token), args))
    return 0


async def _serve(voice_app, args) -> None:
    """Un seul process, deux écoutes.

    L'API mlx-audio reste sur la boucle locale — elle n'a aucune
    authentification, et le workflow a toujours refusé de la publier. La
    voix écoute sur 0.0.0.0 parce que Caddy la joint par
    host.docker.internal, et c'est le jeton qui la garde.

    Les deux partagent le même interpréteur, donc les mêmes modèles en
    mémoire. C'était toute la raison de fusionner les deux services.
    """
    servers = [uvicorn.Server(uvicorn.Config(
        voice_app, host=args.host, port=args.port, log_level="warning"))]
    if not args.no_api:
        from mlx_audio.server import app as api_app
        servers.append(uvicorn.Server(uvicorn.Config(
            api_app, host="127.0.0.1", port=args.api_port, log_level="warning")))
    await asyncio.gather(*(s.serve() for s in servers))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nFini.", file=sys.stderr)
