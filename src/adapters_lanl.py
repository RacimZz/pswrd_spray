from __future__ import annotations

"""Adapter pour le dataset LANL Auth.

Format du fichier décompressé: une ligne par event
    time,user,computer
    1,U1,C1
    1,U1,C2
    ...

- time   = offset en secondes depuis une epoch interne (entier)
- user   = U<id>  (anonymisé)
- computer = C<id> (anonymisé, on l'utilise comme src_ip proxy)
- Seulement des SUCCÈS (pas d'échecs dans ce dataset)

Stratégie pour simuler des échecs détectables:
    -> On ne peut pas détecter du password spray directement (pas d'échecs).
    -> On fait de la détection d'anomalie sur VOLUME et DIVERSITÉ de comptes cibles.
    -> Un "spray" sur ce dataset = 1 user qui se connecte à BEAUCOUP de computers en peu de temps.
    -> Un "bruteforce" = 1 computer ciblé par BEAUCOUP de users différents en peu de temps.

Références:
    https://csr.lanl.gov/data/auth/
    Kent, A.D. (2014). User-Computer Authentication Associations in Time. LANL.
"""

import bz2
import math
from pathlib import Path
from typing import Optional

import pandas as pd


EPOCH_OFFSET = pd.Timestamp("2014-01-01", tz="UTC")
CHUNK_SIZE = 5_000_000  # lignes par chunk


def load_lanl_chunk(
    path: str,
    n_rows: Optional[int] = 30_000_000,
    compressed: bool = True,
) -> pd.DataFrame:
    """Charge un fichier LANL auth (compressé bz2 ou décompressé .txt/.csv).

    Args:
        path: chemin vers le fichier .bz2 ou .txt
        n_rows: nombre max de lignes à charger (None = tout)
        compressed: True si le fichier est .bz2

    Returns:
        DataFrame normalisé avec colonnes: ts, user, src_ip, app, result
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Fichier non trouvé: {path}\n"
            "Télécharge le dataset ici:\n"
            "  http://lanl.ma.ic.ac.uk/data/auth/lanl-auth-dataset-1-00.bz2\n"
            "Puis place le fichier dans data/public/lanl/"
        )

    open_fn = bz2.open if compressed else open

    chunks = []
    rows_read = 0

    with open_fn(path, "rt", encoding="utf-8") as f:
        while True:
            limit = CHUNK_SIZE if n_rows is None else min(CHUNK_SIZE, n_rows - rows_read)
            if limit <= 0:
                break

            chunk = pd.read_csv(
                f,
                header=None,
                names=["time", "user", "computer"],
                nrows=limit,
            )

            if chunk.empty:
                break

            chunks.append(chunk)
            rows_read += len(chunk)

            if n_rows is not None and rows_read >= n_rows:
                break

    if not chunks:
        raise ValueError(f"Fichier vide ou illisible: {path}")

    df = pd.concat(chunks, ignore_index=True)

    # Convertir time (offset en secondes) -> timestamp UTC
    df["ts"] = EPOCH_OFFSET + pd.to_timedelta(df["time"], unit="s")

    # user -> str (U<id>)
    df["user"] = df["user"].astype(str)

    # computer -> src_ip proxy (C<id>)
    df["src_ip"] = df["computer"].astype(str)

    # app = AUTH (fixe, pas d'info dans le dataset)
    df["app"] = "AUTH"

    # Tous des succès (le dataset ne contient que des authentifications réussies)
    df["result"] = "success"

    # Colonnes optionnelles
    df["reason"] = "ok"
    df["user_agent"] = None
    df["country"] = None

    out = df[["ts", "user", "src_ip", "app", "result", "reason", "user_agent", "country"]]
    out = out.sort_values("ts").reset_index(drop=True)

    return out


def compute_features_lanl(df: pd.DataFrame, window: str = "10min") -> pd.DataFrame:
    """Features spécifiques LANL.

    Sur ce dataset (uniquement succès), on détecte:
    - 1 user -> beaucoup de computers en peu de temps (mouvement latéral / credential hopping)
    - 1 computer ciblé par beaucoup de users (serveur très sollicité / anormal)
    """
    from .features import compute_features_fixed_windows

    # Features standards par src_ip (computer) sur fenêtres
    feat_by_computer = compute_features_fixed_windows(df, window=window)
    feat_by_computer["perspective"] = "by_computer"

    # Features par user (en swappant user/src_ip)
    df_swap = df.copy()
    df_swap["src_ip"] = df["user"]
    df_swap["user"] = df["src_ip"]
    feat_by_user = compute_features_fixed_windows(df_swap, window=window)
    feat_by_user = feat_by_user.rename(columns={"src_ip": "actor_user"})
    feat_by_user["perspective"] = "by_user"

    return feat_by_computer, feat_by_user
