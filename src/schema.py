from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


REQUIRED_COLS = ["ts", "user", "src_ip", "app", "result"]
OPTIONAL_COLS = ["reason", "user_agent", "country"]


@dataclass(frozen=True)
class ColumnMap:
    ts: str = "ts"
    user: str = "user"
    src_ip: str = "src_ip"
    app: str = "app"
    result: str = "result"
    reason: Optional[str] = "reason"
    user_agent: Optional[str] = "user_agent"
    country: Optional[str] = "country"

    def to_dict(self) -> Dict[str, str]:
        d = {
            "ts": self.ts,
            "user": self.user,
            "src_ip": self.src_ip,
            "app": self.app,
            "result": self.result,
        }
        if self.reason:
            d["reason"] = self.reason
        if self.user_agent:
            d["user_agent"] = self.user_agent
        if self.country:
            d["country"] = self.country
        return d


def normalize_logs(df: pd.DataFrame, colmap: ColumnMap = ColumnMap()) -> pd.DataFrame:
    """Normalize input logs to the project schema.

    - Renames columns based on colmap
    - Parses timestamps (UTC)
    - Keeps only known columns
    - Ensures result is in {success, fail}
    """
    mapping = {v: k for k, v in colmap.to_dict().items()}
    missing = [colmap.to_dict()[k] for k in ["ts", "user", "src_ip", "app", "result"] if colmap.to_dict()[k] not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input: {missing}")

    df = df.rename(columns=mapping).copy()

    # Ensure all expected columns exist
    for c in OPTIONAL_COLS:
        if c not in df.columns:
            df[c] = None

    # Parse timestamps
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if df["ts"].isna().any():
        bad = int(df["ts"].isna().sum())
        raise ValueError(f"Failed to parse {bad} timestamps in column 'ts'")

    # Normalize result
    df["result"] = df["result"].astype(str).str.lower().str.strip()
    df.loc[df["result"].isin(["ok", "true", "1", "successful", "succeed"]), "result"] = "success"
    df.loc[df["result"].isin(["false", "0", "failed", "failure"]), "result"] = "fail"

    valid = {"success", "fail"}
    bad = df.loc[~df["result"].isin(valid), "result"].unique().tolist()
    if bad:
        raise ValueError(f"Invalid values in 'result': {bad}. Expected: {sorted(valid)}")

    df = df[["ts"] + REQUIRED_COLS[1:] + OPTIONAL_COLS]
    df = df.sort_values("ts").reset_index(drop=True)
    return df

RBA_COLUMN_MAP = ColumnMap(
    ts         = "Login Timestamp",
    user       = "User ID",
    src_ip     = "IP Address",
    app        = "app",          # absent dans RBA, on injecte après
    result     = "Login Successful",
    user_agent = "User Agent String",
    country    = "Country",
)

def normalize_rba(df: pd.DataFrame) -> pd.DataFrame:
    """Injecte la colonne app manquante puis normalise avec le mapping RBA."""
    df = df.copy()
    df["app"] = "rba"
    # Login Successful est booléen → convertir en string lisible par normalize_logs
    df["Login Successful"] = df["Login Successful"].map(
        {True: "success", False: "fail", 1: "success", 0: "fail"}
    )
    return normalize_logs(df, colmap=RBA_COLUMN_MAP)
