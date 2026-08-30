#!/bin/sh
# Pre-install de l'agent vocal : préparer l'hôte AVANT
# `pipx install mlx-audio[server]`.
#
# Trois choses qu'installer le paquet PyPI ne suffit pas à faire :
#
#   1. Vérifier Python et pipx. mlx-audio exige Python 3.10+ ; sans pipx,
#      le moteur n'a aucun moyen d'installer quoi que ce soit.
#   2. Installer espeak-ng. Kokoro convertit le texte en phonèmes via
#      cette bibliothèque C. Le wheel `espeakng-loader` en embarque bien
#      une copie, mais compilée avec le chemin de données de LEUR machine
#      de CI (/Users/runner/work/...) : elle plante à l'usage, et
#      l'override Python de phonemizer n'y change rien. La version
#      Homebrew, elle, connaît ses propres chemins.
#   3. Poser l'unité de service. Un paquet PyPI n'a ni LaunchAgent ni
#      unit systemd ; sans ce fichier il n'y a rien à charger au démarrage.
#
# Idempotent : rejouable sans rien casser.
#
# Sortie stdout = un rapport destiné à être lu (il donne le jeton et les
# chemins). Le bruit des commandes part dans /dev/null.

set -eu

PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"
export PATH

# L'application, à côté de ce script — on la copie dans DATA_DIR pour que
# le service tourne sur une copie stable, indépendante de ce dépôt.
SRC="$(cd "$(dirname "$0")/../app" && pwd)"

# Surchargeables par l'environnement. Le label garde sa valeur historique
# par défaut : le changer orphelinerait le LaunchAgent des installations
# déjà en place, que plus rien ne saurait arrêter.
LABEL="${LABEL:-dev.mlx-audio.flowhub}"
PORT="${PORT:-8131}"
DATA_DIR="${DATA_DIR:-$HOME/.mlx-audio}"
MIN_PY_MINOR=10

# --- Python + pipx ---------------------------------------------------
command -v python3 >/dev/null 2>&1 || {
	echo "✗ python3 est introuvable. Installe Python 3.$MIN_PY_MINOR ou plus récent, puis relance."
	exit 1
}
py_version=$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
py_major=${py_version%%.*}
py_minor=${py_version#*.}
if [ "$py_major" -lt 3 ] || { [ "$py_major" -eq 3 ] && [ "$py_minor" -lt "$MIN_PY_MINOR" ]; }; then
	echo "✗ Python $py_version détecté, mlx-audio exige 3.$MIN_PY_MINOR ou plus."
	exit 1
fi

if ! command -v pipx >/dev/null 2>&1; then
	echo "  pipx absent, installation via Homebrew…"
	command -v brew >/dev/null 2>&1 || {
		echo "✗ Ni pipx ni Homebrew. Installe l'un des deux puis relance."
		exit 1
	}
	brew install pipx >/dev/null 2>&1 || {
		echo "✗ Échec de l'installation de pipx."
		exit 1
	}
	pipx ensurepath >/dev/null 2>&1 || true
fi

# --- espeak-ng -------------------------------------------------------
# Requis seulement par la synthèse ; la transcription s'en passe. On ne
# fait donc pas échouer l'install s'il manque — on le dit, et la dictée
# fonctionnera quand même.
espeak_status="absent — la synthèse vocale sera indisponible (la dictée, elle, marchera)"
ESPEAK_LIB=""
if command -v brew >/dev/null 2>&1; then
	brew list espeak-ng >/dev/null 2>&1 || brew install espeak-ng >/dev/null 2>&1 || true
fi
for candidate in /opt/homebrew/lib/libespeak-ng.dylib /usr/local/lib/libespeak-ng.dylib; do
	if [ -f "$candidate" ]; then
		ESPEAK_LIB="$candidate"
		espeak_status="$candidate"
		break
	fi
done

# --- Chemins absolus -------------------------------------------------
# launchd démarre avec un PATH minimal : tout ce que l'unité invoque doit
# être écrit en absolu.
# On DEMANDE le chemin à pipx plutôt que de le supposer : il est
# configurable (PIPX_BIN_DIR) et diffère selon les installations.
PIPX_BIN=$(pipx environment --value PIPX_BIN_DIR 2>/dev/null || true)
[ -n "$PIPX_BIN" ] || PIPX_BIN="$HOME/.local/bin"
# Le venv n'existe PAS encore — ce script tourne avant `pipx install`. On
# calcule quand même son chemin : le plist qu'on écrit ici ne sera chargé
# qu'après l'installation, au `start:` du workflow.
PIPX_HOME_DIR=$(pipx environment --value PIPX_HOME 2>/dev/null || echo "$HOME/.local/pipx")
VENV_PY="$PIPX_HOME_DIR/venvs/mlx-audio/bin/python"
SERVER_BIN="$PIPX_BIN/mlx_audio.server"
PY_BIN_DIR=$(dirname "$(command -v python3)")

UNIT_PATH=""
path_add() {
	[ -n "$1" ] || return 0
	case ":$UNIT_PATH:" in
	*":$1:"*) return 0 ;;
	esac
	UNIT_PATH="${UNIT_PATH:+$UNIT_PATH:}$1"
}
path_add "$PIPX_BIN"
path_add "$PY_BIN_DIR"
path_add "/opt/homebrew/bin"
path_add "/usr/bin"
path_add "/bin"
path_add "/usr/sbin"
path_add "/sbin"

mkdir -p "$DATA_DIR/hf" "$DATA_DIR/logs"

# --- L'application vocale --------------------------------------------
# Copie plutôt que lien : le store peut être mis à jour ou déplacé sous
# les pieds du service. Une copie fige ce qui tourne.
rm -rf "$DATA_DIR/app"
mkdir -p "$DATA_DIR/app"
cp -R "$SRC/." "$DATA_DIR/app/"

# --- Jeton d'accès de l'agent vocal ----------------------------------
# Généré UNE fois et conservé : le régénérer invaliderait le raccourci
# posé sur l'écran d'accueil.
TOKEN_FILE="$DATA_DIR/token"
if [ ! -s "$TOKEN_FILE" ]; then
	python3 -c "import secrets;print(secrets.token_urlsafe(16))" >"$TOKEN_FILE"
	chmod 600 "$TOKEN_FILE"
	token_status="généré"
else
	token_status="conservé"
fi
TOKEN=$(cat "$TOKEN_FILE")

# --- Espace disque ---------------------------------------------------
# Les poids pèsent ~1,8 Go (Whisper large-v3-turbo + Kokoro + la voix
# d'origine). Sans marge, le téléchargement remplit le disque et laisse un
# cache à moitié écrit que rien ne nettoie. On refuse tôt plutôt que de
# saturer la machine — et on le dit en gigaoctets, pas en pourcentage.
NEEDED_MB=4096
avail_mb=$(df -Pm "$DATA_DIR" | awk 'NR==2 {print $4}')
if [ -n "$avail_mb" ] && [ "$avail_mb" -lt "$NEEDED_MB" ]; then
	echo "✗ Espace disque insuffisant : $((avail_mb / 1024)) Go disponibles."
	echo "  Les modèles pèsent ~1,8 Go et il faut de la marge pour les écrire ;"
	echo "  compte $((NEEDED_MB / 1024)) Go. Fais de la place puis relance."
	exit 1
fi

# --- LaunchAgent -----------------------------------------------------
# Liaison sur 127.0.0.1 et pas 0.0.0.0 : ce serveur n'a aucune
# authentification, il ne doit pas être joignable depuis le LAN. Les
# applications distantes passent par leur propre proxy.
AGENT_DIR="$HOME/Library/LaunchAgents"
PLIST="$AGENT_DIR/$LABEL.plist"
mkdir -p "$AGENT_DIR"

espeak_env=""
if [ -n "$ESPEAK_LIB" ]; then
	espeak_env="    <key>PHONEMIZER_ESPEAK_LIBRARY</key><string>$ESPEAK_LIB</string>"
fi

cat >"$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<!-- Généré par scripts/prepare-mlx-audio.sh — rejouable. -->
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <!-- Un seul process pour les deux besoins : l'API de dictée sur la
         boucle locale, l'agent vocal sur 0.0.0.0 derrière son jeton. Les
         modèles ne sont chargés qu'une fois et servent aux deux. -->
    <string>$VENV_PY</string>
    <string>-u</string>
    <string>$DATA_DIR/app/server.py</string>
    <string>--host</string><string>0.0.0.0</string>
    <string>--port</string><string>8140</string>
    <string>--api-port</string><string>$PORT</string>
  </array>
  <!-- Filet de sécurité, et désormais la SEULE protection : mlx-audio
       peut faire mkdir("logs") relatif au répertoire courant. Sous launchd
       ce répertoire est /, en lecture seule — le serveur mourrait sur
       "Read-only file system: 'logs'". On lui donne un cwd inscriptible.
       (Le drapeau --log-dir n'existe plus : il appartenait à l'entrée CLI
       de mlx_audio.server, que nous n'utilisons plus — nous importons son
       application FastAPI.) -->
  <key>WorkingDirectory</key><string>$DATA_DIR</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>$UNIT_PATH</string>
    <key>HF_HOME</key><string>$DATA_DIR/hf</string>
    <key>VOICE_AGENT_TOKEN</key><string>$TOKEN</string>
    <key>HOME</key><string>$HOME</string>
$espeak_env
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$DATA_DIR/mlx-audio.log</string>
  <key>StandardErrorPath</key><string>$DATA_DIR/mlx-audio.log</string>
</dict>
</plist>
EOF
chmod 644 "$PLIST"

# --- Rapport ---------------------------------------------------------
echo "Python : $py_version"
echo "espeak-ng : $espeak_status"
echo "LaunchAgent : $PLIST"
echo "Application vocale : $DATA_DIR/app"
echo "Jeton de l'agent vocal : $token_status → $DATA_DIR/token"
echo "API de dictée : 127.0.0.1:$PORT (locale — pas d'authentification)"
echo "Agent vocal : 0.0.0.0:8140 (publié par l'ingress, protégé par jeton)"
echo "Cache modèles : $DATA_DIR/hf"
