"""Synthèse française, phrase par phrase.

Remplace le PocketTTSResponder du pipeline. Trois raisons, dans l'ordre
d'importance :

1. L'ACCENT. Pocket TTS n'a que des voix anglophones ; « cosette » lit du
   français avec les phonèmes de l'anglais. Kokoro a `ff_siwis`, une voix
   française — mais il faut AUSSI lui passer `lang_code="f"`, sinon il
   charge la voix dans le pipeline anglais et prévient poliment
   (« Language mismatch, loading ff_siwis voice into American English
   pipeline ») avant de produire exactement le même accent.

2. LA TRONCATURE. Kokoro avertit lui-même : « Chunking logic not yet
   implemented, so long texts may be truncated unless you split them ».
   Une réponse un peu longue serait coupée en silence. On découpe donc.

3. LA LATENCE. Le pipeline consomme ce générateur bloc par bloc et joue
   ce qu'il reçoit. En rendant la première phrase tout de suite, on parle
   pendant que les suivantes se calculent, au lieu d'attendre la réponse
   entière. C'est le même raisonnement que le streaming du LLM, appliqué
   à la sortie.

Mesuré sur M4, voix ff_siwis à chaud : 4,55 s d'audio produites en 0,33 s,
soit ×14 le temps réel. Le premier appel coûte 1,5 s de plus (création du
pipeline espeak), d'où le préchauffage à l'initialisation.
"""

import re
import sys
import time
from typing import Any, Iterable, Optional

# On coupe après . ! ? … suivis d'un espace ou de la fin. Le lookbehind
# écarte les décimales (« 3.5 ») et les abréviations d'une lettre
# (« M. Dupont »), qui produiraient des fragments absurdes à l'oral.
_SPLIT = re.compile(r"(?<![0-9A-ZÀ-Ý])(?<=[.!?…])\s+")

# En dessous, un fragment ne vaut pas un appel : le surcoût fixe de la
# synthèse dépasse ce qu'on gagne à le rendre plus tôt.
MIN_CHARS = 40
# Sauf pour le PREMIER, qui est un cas à part : c'est le seul dont
# l'utilisateur attend l'arrivée en silence. Tous les suivants se
# calculent pendant que le précédent se joue, donc leur coût est masqué.
# On accepte donc un morceau court en tête — quitte à ce que la césure
# soit moins naturelle — parce qu'il transforme le ressenti.
MIN_CHARS_FIRST = 6
# Au-dessus, on refend sur la virgule : c'est la limite où la troncature
# d'espeak commence à se manifester.
MAX_CHARS = 220


# On coupe aussi le tout premier morceau sur une virgule ou un tiret, pas
# seulement sur une fin de phrase : « Bonjour, je vais bien » doit pouvoir
# rendre « Bonjour, » immédiatement.
_FIRST_CUT = re.compile(r"(?<=[,;:—–])\s+")


def split_sentences(text: str) -> list:
    """Découpe en unités prononçables, ni trop courtes ni trop longues."""
    text = text.strip()

    head = ""
    m = _FIRST_CUT.search(text)
    if m and MIN_CHARS_FIRST <= m.start() and len(text) - m.end() >= MIN_CHARS:
        head, text = text[:m.start()], text[m.end():]

    parts, buffer = [], ""
    for piece in _SPLIT.split(text):
        if not piece:
            continue
        buffer = f"{buffer} {piece}".strip() if buffer else piece
        if len(buffer) >= MIN_CHARS:
            parts.append(buffer)
            buffer = ""
    if buffer:
        # Le reliquat rejoint la phrase précédente plutôt que de partir
        # seul : « Merci. » tout court sonne détaché.
        if parts:
            parts[-1] = f"{parts[-1]} {buffer}"
        else:
            parts.append(buffer)

    out = [head] if head else []
    for part in parts:
        while len(part) > MAX_CHARS:
            cut = part.rfind(",", 0, MAX_CHARS)
            if cut <= 0:
                cut = part.rfind(" ", 0, MAX_CHARS)
            if cut <= 0:
                break
            out.append(part[:cut + 1].strip())
            part = part[cut + 1:].strip()
        if part:
            out.append(part)
    return out


class FrenchKokoroResponder:
    """Même interface que PocketTTSResponder : sample_rate + create_generator."""

    def __init__(self, model, *, voice: str = "ff_siwis", lang_code: str = "f",
                 streaming_interval: float = 0.32,
                 temperature: Optional[float] = None):
        self.model = model
        self.voice = voice
        self.lang_code = lang_code
        self.streaming_interval = streaming_interval
        self.temperature = temperature
        self.sample_rate = int(getattr(model, "sample_rate", 24_000) or 24_000)

    def _kwargs(self, text: str) -> dict:
        kwargs = {
            "text": text,
            "voice": self.voice,
            # Sans lang_code, Kokoro retombe sur 'a' (anglais américain)
            # quelle que soit la voix. C'est LA ligne qui enlève l'accent.
            "lang_code": self.lang_code,
            "stream": True,
            "streaming_interval": self.streaming_interval,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        return kwargs

    def warmup(self) -> None:
        """Première synthèse à l'installation plutôt qu'au premier « bonjour ».
        Kokoro construit son pipeline espeak au premier appel : 1,5 s qui
        seraient sinon payées par l'utilisateur, en tête de conversation,
        là où l'attente se remarque le plus."""
        t0 = time.monotonic()
        try:
            for _ in self.model.generate(**self._kwargs("Bonjour.")):
                break
        except Exception as e:
            # Un préchauffage raté ne doit rien empêcher, mais il doit se
            # VOIR : sinon la première réponse est lente sans raison
            # apparente et on cherche le problème ailleurs.
            print(f"  ! préchauffage TTS en échec : {type(e).__name__}: {e}",
                  file=sys.stderr)
            return
        print(f"  préchauffage TTS : {time.monotonic() - t0:.2f}s", file=sys.stderr)

    def create_generator(self, text: str) -> Iterable[Any]:
        for sentence in split_sentences(text) or [text]:
            t0 = time.monotonic()
            for seg in self.model.generate(**self._kwargs(sentence)):
                print(f"  tts {len(sentence):3d} car. -> {time.monotonic() - t0:5.2f}s",
                      file=sys.stderr)
                yield seg
                t0 = time.monotonic()
