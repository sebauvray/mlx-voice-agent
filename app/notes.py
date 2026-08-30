"""La mémoire de la secrétaire : un fichier par session, écrit au fil de l'eau.

Sans ça, la conversation ne vit que dans `_conversation`, en mémoire, et
disparaît à la fermeture de l'onglet. Une secrétaire qui perd ses notes en
sortant du bureau ne sert à rien.

Écrit APRÈS CHAQUE TOUR, pas à la fin : un plantage, une déconnexion ou un
redémarrage du service ne doit pas coûter la conversation. Le fichier est
lisible pendant qu'on parle.

Format Markdown, parce que la note est destinée à être relue par un humain
et reprise par un autre outil — la passe de mise en forme viendra plus
tard, et elle partira de là.
"""

import datetime
import pathlib
import re
import sys
from typing import Optional

DEFAULT_DIR = pathlib.Path.home() / ".mlx-audio" / "notes"


def _slug(text: str, limit: int = 48) -> str:
    """Un titre de fichier tiré des premiers mots dits. Bien plus utile
    qu'un horodatage seul quand on cherche « la note sur la musique »."""
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:limit].strip("-") or "note"


class Notebook:
    def __init__(self, directory: Optional[pathlib.Path] = None):
        self.dir = pathlib.Path(directory) if directory else DEFAULT_DIR
        self.path: Optional[pathlib.Path] = None
        self._turns = 0

    def open(self) -> None:
        """Appelé à l'ouverture d'une session. Le fichier n'est PAS créé
        ici : une session ouverte puis fermée sans un mot ne doit pas
        laisser de note vide derrière elle."""
        self.path = None
        self._turns = 0

    def add(self, spoken: str, reply: str) -> None:
        if not spoken.strip():
            return
        try:
            if self.path is None:
                self.dir.mkdir(parents=True, exist_ok=True)
                now = datetime.datetime.now()
                # Le nom se fige sur la PREMIÈRE phrase : c'est en général
                # celle qui annonce le sujet.
                self.path = self.dir / f"{now:%Y-%m-%d_%H%M}-{_slug(spoken)}.md"
                self.path.write_text(
                    f"# Note vocale — {now:%d/%m/%Y %H:%M}\n\n", encoding="utf-8"
                )
                print(f"  ✎ note : {self.path}", file=sys.stderr)
            self._turns += 1
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(f"**Moi.** {spoken.strip()}\n\n")
                if reply.strip():
                    fh.write(f"> {reply.strip()}\n\n")
        except OSError as e:
            # Une note qu'on n'arrive pas à écrire ne doit pas couper la
            # conversation — mais elle doit se voir.
            print(f"  ! note non écrite : {type(e).__name__}: {e}", file=sys.stderr)

    def close(self) -> Optional[pathlib.Path]:
        if self.path is not None:
            print(f"  ✎ note close ({self._turns} tours) : {self.path}",
                  file=sys.stderr)
        return self.path
