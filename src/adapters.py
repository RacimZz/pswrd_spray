from __future__ import annotations

"""Dataset adapters.

But: mapper des datasets publics vers le schéma standard (ts,user,src_ip,app,result,...).

Tu complèteras ces fonctions quand tu auras téléchargé les données.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .schema import ColumnMap, normalize_logs


def adapter_generic_csv(path: str, colmap: ColumnMap) -> pd.DataFrame:
    df = pd.read_csv(path)
    return normalize_logs(df, colmap=colmap)


def adapter_lanl_auth(path: str) -> pd.DataFrame:
    """LANL auth dataset (souvent: time,user,computer,...) -> notre schéma.

    NOTE: pas d'IP réelle; on peut utiliser 'computer' comme src_ip proxy, et fixer app='AUTH'.
    """
    df = pd.read_csv(path)

    # À ADAPTER selon le fichier exact (tu me l'enverras et on ajustera)
    candidates = df.columns.tolist()
    raise NotImplementedError(
        f"Adapter LANL non configuré. Colonnes trouvées: {candidates}. "
        "Dis-moi les noms exacts de colonnes (timestamp/user/src) et je te donne le mapping."
    )
