#!/usr/bin/env python3
"""Épreuve de bout en bout du serveur, sans navigateur ni micro.

On ne peut pas piloter getUserMedia depuis un terminal, mais tout ce qui
est en amont se teste : on remplace le micro par un WAV fabriqué avec la
voix système et le LLM par un faux endpoint. Si ça passe, il ne reste
comme inconnue que la capture navigateur.

Le WAV est injecté À PEU PRÈS EN TEMPS RÉEL. Le gaver d'un coup ne
testerait rien : le VAD et smart-turn raisonnent sur des durées, un
fichier envoyé en 50 ms ressemble à un hoquet, pas à une phrase.
"""
import asyncio, json, os, pathlib, subprocess, sys, threading, time, wave

APP = pathlib.Path(__file__).resolve().parent.parent / "app"
from http.server import BaseHTTPRequestHandler, HTTPServer

import websockets

LLM_PORT, APP_PORT = 8141, 8142
TOKEN = "selftest-token"        # le serveur exige un jeton depuis qu'il est exposable
PHRASE = os.environ.get("SELFTEST_PHRASE") or "Bonjour, peux-tu me dire quelle heure il est s'il te plaît ?"
REPLY = "Il est bientôt midi. Bonne journée."


class FakeLLM(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.dumps({"choices": [{"message": {"role": "assistant",
                                                    "content": REPLY}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


async def run() -> int:
    # --real : on parle au LLM de l'infra, bail sur la carte compris. Sans
    # ce drapeau, un faux endpoint suffit pour éprouver la chaîne audio
    # sans consommer de GPU.
    real = "--real" in sys.argv
    srv = None
    if not real:
        srv = HTTPServer(("127.0.0.1", LLM_PORT), FakeLLM)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"  faux LLM sur :{LLM_PORT}")
    else:
        print("  LLM RÉEL (conf thebureau) + réservation de carte")

    subprocess.run(["say", "-o", "/tmp/vt.wav", "--data-format=LEI16@16000",
                    "--channels=1", PHRASE], check=True)
    with wave.open("/tmp/vt.wav") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        pcm = w.readframes(w.getnframes())
    print(f"  échantillon : {len(pcm)/2/16000:.1f} s — « {PHRASE} »")

    # --no-api : le harnais n'éprouve que la voix, et sans ce drapeau il
    # tenterait de lier le 8131 déjà pris par le service installé.
    cmd = [sys.executable, str(APP / "server.py"), "--port", str(APP_PORT), "--verbose",
           "--token", TOKEN, "--no-api"]
    if not real:
        cmd += ["--llm-url", f"http://127.0.0.1:{LLM_PORT}/v1/chat/completions",
                "--llm-model", "fake", "--no-lease"]
    app = subprocess.Popen(
        cmd,
        env={**os.environ, "HF_HOME": os.path.expanduser("~/.mlx-audio/hf")},
        text=True)  # sortie héritée : on veut voir ses erreurs telles quelles

    got = {"partial": "", "exchange": None, "audio": 0, "chunks": 0,
           "states": [], "events": [], "t_first_audio": None, "t_exchange": None}
    try:
        for _ in range(60):
            if app.poll() is not None:
                print("✗ le serveur s'est arrêté (voir ci-dessus)")
                return 1
            try:
                ws = await websockets.connect(
                    f"ws://127.0.0.1:{APP_PORT}/ws?k={TOKEN}")
                break
            except Exception:
                await asyncio.sleep(1)
        else:
            print("✗ serveur injoignable"); return 1

        async with ws:
            t0 = time.monotonic()
            print("  chargement des modèles…", flush=True)
            while True:                                   # attendre "ready"
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=300))
                if m["type"] == "ready":
                    print(f"  prêt en {time.monotonic()-t0:.0f} s "
                          f"(sortie {m['sample_rate']} Hz)")
                    break
                if m["type"] in ("busy", "fatal"):
                    print(f"✗ {m['text']}"); return 1
                if m["type"] == "note":
                    print(f"  · {m['text']}")

            async def collect():
                while True:
                    msg = await ws.recv()
                    if isinstance(msg, bytes):
                        if got["t_first_audio"] is None:
                            got["t_first_audio"] = time.monotonic()
                        got["audio"] += len(msg); got["chunks"] += 1
                        continue
                    m = json.loads(msg)
                    if m["type"] == "partial":  got["partial"] += m["text"]
                    elif m["type"] == "state":  got["states"].append(m["value"])
                    elif m["type"] == "exchange":
                        got["exchange"] = m
                        got["t_exchange"] = time.monotonic()
                    elif m["type"] == "note":   print(f"  · {m['text']}")
                    elif m["type"] == "dbg":
                        got["events"].append(m["event"])
                        print(f"    [{m['event']}] {m['fields']}")
                    elif m["type"] in ("error", "fatal"): print(f"  ! {m['text']}")

            task = asyncio.create_task(collect())
            FRAME = 512 * 2                               # 32 ms en octets
            spoken = time.monotonic()
            for i in range(0, len(pcm), FRAME):
                await ws.send(pcm[i:i + FRAME])
                await asyncio.sleep(0.032)
            t_speech_end = time.monotonic()
            print(f"  phrase envoyée en {t_speech_end-spoken:.1f} s, "
                  f"puis silence pour déclencher la fin de tour")
            silence = b"\x00" * FRAME
            # 45 s : le PREMIER appel TTS inclut le préchauffage du modèle,
            # bien plus lent que les suivants. Couper trop tôt fait conclure
            # à une panne là où il n'y a qu'une compilation.
            mute = os.environ.get("SELFTEST_MUTE_AFTER_TURN") == "1"
            for _ in range(int(45 / 0.032)):
                # SELFTEST_MUTE_AFTER_TURN=1 : on cesse d'alimenter le VAD
                # dès le tour clos, pour isoler la contention sur le thread
                # MLX pendant la synthèse.
                if not (mute and got["exchange"]):
                    await ws.send(silence)
                await asyncio.sleep(0.032)
                if got["exchange"] and got["chunks"]:
                    await asyncio.sleep(2.0)   # laisser arriver la suite
                    break
            task.cancel()
    finally:
        app.terminate()
        try:
            app.wait(timeout=10)
        except subprocess.TimeoutExpired:
            app.kill()
        if srv is not None:
            srv.shutdown()

    print("\n" + "=" * 58)
    ok = True
    def check(label, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'OK ' if cond else 'ÉCHEC'}  {label:22s} {detail}")

    check("transcription live", got["partial"].strip(), f"« {got['partial'].strip()} »")
    check("tour complet + LLM", got["exchange"],
          f"« {got['exchange']['user']} » → {got['exchange']['llm_ms']} ms"
          if got["exchange"] else "aucun échange reçu")
    if got["exchange"]:
        print(f"         réponse : « {got['exchange']['assistant']} »")
    check("audio TTS renvoyé", got["chunks"],
          f"{got['chunks']} blocs, {got['audio']/2/24000:.1f} s de parole")
    if got["t_exchange"]:
        print(f"\n  Latence depuis la fin de parole :")
        print(f"    STT final + LLM  {got['t_exchange']-t_speech_end:6.2f} s"
              f"   (dont LLM {got['exchange']['llm_ms']/1000:.2f} s)")
        if got["t_first_audio"]:
            print(f"    premier son      {got['t_first_audio']-t_speech_end:6.2f} s"
                  f"   ← l'attente réellement ressentie")
    check("machine à états", "speaking" in got["states"], "→".join(got["states"][:6]))
    print("=" * 58)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
