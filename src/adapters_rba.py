from __future__ import annotations

"""Adapter pour le dataset RBA (Risk-Based Authentication).

Source: https://github.com/das-group/rba-dataset
Download: https://github.com/das-group/rba-dataset/releases (rba-dataset.zip, ~1GB)

Colonnes du dataset:
    IP Address, Country, Region, City, ASN, UserAgent (OS Name/Version, Browser, Device),
    User ID, Login Timestamp (64-bit), Round-Trip Time, Login Successful,
    Is Attack IP, Is Account Takeover

C'est le dataset le plus "riche" pour notre projet:
- Login Successful (True/False) = success/fail
- Is Attack IP = label pour évaluer notre détection
- Is Account Takeover = signal de compromission (notre success_after_fail)
- IP Address = src_ip direct
- User ID = user
- Login Timestamp = ts (Unix ms)
"""

from pathlib import Path
from typing import Optional

import pandas as pd


RBA_COL_MAP = {
    "IP Address": "src_ip",
    "User ID": "user",
    "Login Timestamp": "ts",
    "Login Successful": "result",
    "Country": "country",
    "Round-Trip Time (ms)": "rtt",
    "Is Attack IP": "is_attack_ip",
    "Is Account Takeover": "is_account_takeover",
}


def load_rba(path: str, n_rows: Optional[int] = None) -> pd.DataFrame:
    """Charge et normalise le dataset RBA.

    Args:
        path: chemin vers le CSV principal du dataset
        n_rows: nombre max de lignes (None = tout)

    Returns:
        DataFrame normalisé + colonnes label (is_attack_ip, is_account_takeover)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Fichier non trouvé: {path}\n"
            "Télécharge le dataset ici:\n"
            "  https://github.com/das-group/rba-dataset/releases\n"
            "Dézippe et place les CSV dans data/public/rba/"
        )

    df = pd.read_csv(path, nrows=n_rows)

    # Rename colonnes connues
    rename = {k: v for k, v in RBA_COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Timestamp: Unix ms -> datetime UTC
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True, errors="coerce")
        df = df.dropna(subset=["ts"])

    # result: True/False -> success/fail
    if "result" in df.columns:
        df["result"] = df["result"].astype(str).str.lower().map(
            {"true": "success", "false": "fail", "1": "success", "0": "fail"}
        ).fillna("fail")

    # user_agent: on peut l'extraire de "Browser Name and Version" si présent
    if "Browser Name and Version" in df.columns:
        df["user_agent"] = df["Browser Name and Version"].astype(str)
    elif "user_agent" not in df.columns:
        df["user_agent"] = None

    # app = SSO (ce dataset est SSO)
    df["app"] = "SSO"

    # reason (optionnel)
    df["reason"] = None

    # Labels (garde-les pour évaluation)
    for col in ["is_attack_ip", "is_account_takeover"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map(
                {"true": 1, "false": 0, "1": 1, "0": 0}
            ).fillna(0).astype(int)
        else:
            df[col] = 0

    # Colonnes standard
    base_cols = ["ts", "user", "src_ip", "app", "result", "reason", "user_agent", "country"]
    label_cols = ["is_attack_ip", "is_account_takeover"]

    for c in base_cols:
        if c not in df.columns:
            df[c] = None

    out = df[base_cols + label_cols].sort_values("ts").reset_index(drop=True)
    return out


def evaluate_rba(scored: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Évalue les alertes vs les labels réels du dataset RBA.

    Args:
        scored: DataFrame avec colonnes src_ip, ts, is_anomaly, anomaly_score
        labels: DataFrame original avec is_attack_ip, is_account_takeover

    Returns:
        dict avec métriques: precision, recall, F1, etc.
    """
    from sklearn.metrics import classification_report, confusion_matrix

    # On agrège les labels par (src_ip, window)
    # is_attack_ip: 1 si au moins 1 ligne de la fenêtre est une IP attaque
    labels_agg = (
        labels.set_index("ts")
              .groupby("src_ip")
              .resample("10min")
              .agg(has_attack_ip=("is_attack_ip", "max"), has_takeover=("is_account_takeover", "max"))
              .reset_index()
    )

    merged = scored.merge(labels_agg, on=["src_ip", "ts"], how="left")
    merged["has_attack_ip"] = merged["has_attack_ip"].fillna(0).astype(int)
    merged["has_takeover"] = merged["has_takeover"].fillna(0).astype(int)

    # Évaluation vs attack_ip
    y_true = merged["has_attack_ip"].values
    y_pred = merged["is_anomaly"].values

    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "vs_attack_ip": report,
        "confusion_matrix": cm,
        "n_attack_ips": int(merged["has_attack_ip"].sum()),
        "n_anomalies": int(merged["is_anomaly"].sum()),
    }
