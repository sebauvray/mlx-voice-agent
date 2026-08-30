#!/bin/sh
# Post-install de MLX Audio : compléter le venv, puis pré-télécharger les
# modèles.
#
# ── Pourquoi compléter le venv ────────────────────────────────────────
# `mlx-audio[server]` ne déclare pas tout ce dont Kokoro a besoin. Chaîne
# constatée, chaque manque n'apparaissant qu'une fois le précédent levé :
#
#   misaki            absent de l'extra [server]
#   misaki[en]        num2words, spaCy… absents à leur tour
#   modèle spaCy      misaki le télécharge À L'EXÉCUTION via
#                     spacy.cli.download, qui délègue à `uv pip install`
#                     et échoue faute de trouver le venv depuis le
#                     process serveur — la requête part alors en timeout
#
# Aggravant : quand ça échoue, mlx-audio répond HTTP 200 avec un corps
# VIDE. Rien ne signale l'erreur côté client. D'où le fait de tout poser
# ici, pendant l'install, plutôt que de le découvrir à l'usage.
#
# ── Pourquoi pré-télécharger ──────────────────────────────────────────
# Les modèles arrivent au premier appel. Kokoro tire 56 fichiers, et sans
# jeton Hugging Face le débit est bridé : observé bloqué à 55/56 pendant
# plus de dix minutes. Un utilisateur qui clique sur le micro et attend
# dix minutes conclut, à raison, que ça ne marche pas.
#
# Idempotent : les injections et les téléchargements déjà faits sont des
# no-ops.

set -eu

PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"
export PATH

LABEL="${LABEL:-dev.mlx-audio.flowhub}"
BASE="${MLX_AUDIO_BASE:-http://127.0.0.1:8131}"
DATA_DIR="$HOME/.mlx-audio"
export HF_HOME="$DATA_DIR/hf"

STT_MODEL="mlx-community/whisper-large-v3-turbo-asr-fp16"
TTS_MODEL="mlx-community/Kokoro-82M-bf16"
# Kokoro charge aussi le dépôt d'origine, pas seulement la conversion MLX.
TTS_MODEL_BASE="prince-canuma/Kokoro-82M"
SPACY_WHEEL="https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

# --- Le venv que pipx vient de créer ---------------------------------
PIPX_HOME=$(pipx environment --value PIPX_HOME 2>/dev/null || echo "$HOME/.local/pipx")
VENV_PY="$PIPX_HOME/venvs/mlx-audio/bin/python"
[ -x "$VENV_PY" ] || {
	echo "✗ Environnement pipx de mlx-audio introuvable ($VENV_PY)."
	echo "  L'installation a-t-elle abouti ?"
	exit 1
}

# --- Dépendances de la synthèse --------------------------------------
# Non bloquantes : sans elles la dictée fonctionne, seule la synthèse est
# indisponible. On préfère une install partielle annoncée à un échec sec.
# sentencepiece : requis par le tokenizer de Pocket TTS et non déclaré par
# l'extra [server]. Constaté en usage : sans lui le serveur démarre,
# transcrit, appelle le LLM… puis meurt au moment de parler.
sentencepiece_status="déjà présent"
if ! "$VENV_PY" -c 'import sentencepiece' >/dev/null 2>&1; then
	pipx inject mlx-audio sentencepiece >/dev/null 2>&1 &&
		sentencepiece_status="installé" ||
		sentencepiece_status="ÉCHEC — la synthèse ne fonctionnera pas"
fi

tts_status="prête"
inject() {
	pipx inject mlx-audio "$1" >/dev/null 2>&1 || {
		tts_status="incomplète — synthèse indisponible (échec sur $2)"
		return 1
	}
}
"$VENV_PY" -c 'import misaki.en' >/dev/null 2>&1 || inject "misaki[en]" "misaki" || true
"$VENV_PY" -c 'import spacy,sys; sys.exit(0 if spacy.util.is_package("en_core_web_sm") else 1)' >/dev/null 2>&1 ||
	inject "$SPACY_WHEEL" "modèle spaCy" || true

# --- Pré-téléchargement des modèles ----------------------------------
# snapshot_download reprend là où il s'est arrêté : on peut donc réessayer
# sans repartir de zéro. Trois tentatives, puis on abandonne CE modèle
# sans faire échouer l'install — le service démarrera, le premier appel
# terminera le travail.
fetch() { # $1 = dépôt, $2 = libellé
	n=0
	while [ "$n" -lt 3 ]; do
		if "$VENV_PY" - "$1" <<'PY' >/dev/null 2>&1
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], max_workers=4)
PY
		then
			echo "  $2 : téléchargé"
			return 0
		fi
		n=$((n + 1))
	done
	echo "  $2 : échec après 3 tentatives (se terminera au premier usage)"
	return 1
}

# La conversation utilise d'AUTRES modèles que la dictée : Voxtral
# transcrit en flux (Whisper ne sait pas), smart-turn décide de la fin de
# tour, silero détecte la parole. Les tirer ici évite d'attendre 3,4 Go au
# premier « bonjour ».
VOICE_MODELS="mlx-community/silero-vad mlx-community/smart-turn-v3 mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"

echo "Modèles :"
fetch "$STT_MODEL" "transcription (Whisper large-v3-turbo)" || true
for repo in $VOICE_MODELS; do
	fetch "$repo" "conversation (${repo##*/})" || true
done
if [ "$tts_status" = "prête" ]; then
	fetch "$TTS_MODEL" "synthèse (Kokoro)" || true
	fetch "$TTS_MODEL_BASE" "synthèse (voix)" || true
fi

# --- Redémarrage + vérification --------------------------------------
# Le service a démarré avant ce script, donc sans les injections. On le
# recharge, puis on vérifie pour de vrai plutôt que d'annoncer un succès.
launchctl kickstart -k "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || true

i=0
while [ "$i" -lt 90 ]; do
	curl -sf -o /dev/null "$BASE/docs" 2>/dev/null && break
	i=$((i + 1))
	sleep 1
done
if [ "$i" -ge 90 ]; then
	echo ""
	echo "✗ Le serveur n'a pas répondu sur $BASE au bout de 90 s."
	echo "  Journal : $DATA_DIR/mlx-audio.log"
	exit 1
fi

# Épreuve réelle de la dictée : on fabrique un échantillon avec la voix
# système et on vérifie que la transcription renvoie du texte. `say` écrit
# directement du WAV 16 kHz mono, donc pas besoin de ffmpeg.
stt_status="non vérifiée"
SAMPLE="$DATA_DIR/.selftest.wav"
if say -o "$SAMPLE" --data-format=LEI16@16000 --channels=1 "Bonjour, ceci est un test." >/dev/null 2>&1; then
	got=$(curl -sf -m 300 -X POST "$BASE/v1/audio/transcriptions" \
		-F "file=@$SAMPLE" -F "model=$STT_MODEL" 2>/dev/null |
		"$VENV_PY" -c 'import json,sys; print((json.load(sys.stdin).get("text") or "").strip())' 2>/dev/null || echo '')
	if [ -n "$got" ]; then
		stt_status="OK — \"$got\""
	else
		stt_status="ÉCHEC (voir $DATA_DIR/mlx-audio.log)"
	fi
	rm -f "$SAMPLE"
fi

echo ""
echo "Dictée : $stt_status"
echo "Synthèse : $tts_status"
echo "sentencepiece : $sentencepiece_status"
echo "Cache modèles : $(du -sh "$HF_HOME" 2>/dev/null | cut -f1)"
