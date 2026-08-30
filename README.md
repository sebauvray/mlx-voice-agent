# mlx-voice-agent

Agent conversationnel vocal servi en web depuis un Mac, accéléré par le GPU
Apple. Le micro est là où tu es (téléphone, portable), le GPU est sur la
machine.

Construit **au-dessus** du paquet [`mlx-audio`](https://github.com/Blaizzy/mlx-audio) :
STT streaming, VAD, détection de fin de tour, TTS et barge-in restent ceux
d'amont. Ce dépôt ajoute la couche web (WebSocket audio bidirectionnelle,
page cliente), un moteur LLM branchable et la mise en service macOS.

Le même process sert deux choses :

- l'**API locale** compatible OpenAI (`/v1/audio/transcriptions`,
  `/v1/audio/speech`), sur `127.0.0.1:8131` ;
- l'**agent vocal** web sur `:8140`, protégé par un jeton.

## Installation

```sh
./scripts/prepare-mlx-audio.sh   # vérifs Python/pipx, espeak-ng, LaunchAgent
pipx install "mlx-audio[server]==0.5.0"
./scripts/setup-mlx-audio.sh     # dépendances Python restantes + modèles
```

Les deux scripts sont **idempotents**. `prepare` imprime le jeton de
l'agent vocal et les chemins ; `setup` pré-télécharge les modèles (56
fichiers, plusieurs minutes — le faire ici plutôt qu'au premier appel évite
de conclure que la dictée ne marche pas).

### Pourquoi espeak-ng vient de Homebrew

Kokoro convertit le texte en phonèmes via cette bibliothèque C. Le wheel
`espeakng-loader` en embarque une copie, mais compilée avec le chemin de
données de la machine de CI d'amont (`/Users/runner/work/…`) : elle plante
à l'usage, et l'override Python de phonemizer n'y change rien. La version
Homebrew connaît ses propres chemins.

### Variables

| Variable | Défaut | Rôle |
|---|---|---|
| `LABEL` | `dev.mlx-audio.flowhub` | Label du LaunchAgent. |
| `PORT` | `8131` | Port de l'API locale. |
| `DATA_DIR` | `~/.mlx-audio` | App déployée, cache modèles, logs, jeton. |

Le défaut de `LABEL` garde sa valeur historique : le changer orphelinerait
le LaunchAgent des installations déjà en place, que plus rien ne saurait
arrêter.

## Service

```sh
launchctl bootout   "gui/$(id -u)/dev.mlx-audio.flowhub"
launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/dev.mlx-audio.flowhub.plist
```

`launchd` met la définition du job **en cache** : `kickstart` relance le
processus mais ne relit pas le plist. Sans `bootout` préalable, toute
modification du plist est ignorée et le service redémarre à l'identique.

## Mesures (Mac mini M4 / 16 Go)

| | |
|---|---|
| Transcription | 1,7 s pour 5,5 s d'audio (≈ 3,3× le temps réel) |
| Synthèse | 0,43 s pour 5,7 s produites (≈ 11× le temps réel) |
| Mémoire | 1,7 Go avec le seul STT, 2,9 Go les deux modèles chargés |

La mémoire se mesure au **`footprint`**, pas à la RSS : MLX alloue sur le
tas unifié, que `ps` ne compte pas.

## Sécurité

L'API locale n'a **aucune authentification** et n'écoute que sur la
loopback — elle n'a rien à faire sur le réseau. L'agent vocal, lui, écoute
sur `0.0.0.0` (un navigateur distant doit l'atteindre) et se protège par un
jeton, parce que le port serait sinon joignable sur tout le LAN.

HTTPS est **obligatoire, pas confortable** : `getUserMedia` refuse le micro
hors contexte sécurisé, et une IP privée en clair n'en est pas un. Mets un
reverse proxy avec un vrai certificat devant.

## Tests

```sh
python3 dev/selftest.py
```

## Avec FlowHub

Le workflow `mlx-audio` référence ce dépôt et joue les deux scripts :

```yaml
sources:
  app:
    url: "https://github.com/sebauvray/mlx-voice-agent.git"

pre_install:
  - run.script:
      script: "${sources.app}/scripts/prepare-mlx-audio.sh"
```

Le dépôt reste utilisable seul : FlowHub ne fait que le cloner et lancer
les scripts.
