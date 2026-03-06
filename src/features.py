from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


def _entropy_from_counts(counts: pd.Series) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = (counts / total).to_numpy(dtype=float)
    return float(-(p * np.log2(p + 1e-12)).sum())


def compute_features_fixed_windows(
    logs: pd.DataFrame,
    window: str = "10min",
    by_app: bool = False,
    min_fail_for_success_after_fail: int = 20,
) -> pd.DataFrame:
    df = logs.copy()
    df["is_fail"]    = (df["result"] == "fail").astype(np.int8)
    df["is_success"] = (df["result"] == "success").astype(np.int8)

    # Crée une clé de fenêtre temporelle directement → évite le resample groupby imbriqué
    win_ns  = int(pd.to_timedelta(window).total_seconds()) 
    df["ts_unix"] = df["ts"].astype(np.int64) // 10**9
    df["window"]  = (df["ts_unix"] // win_ns) * win_ns
    df["window"]  = pd.to_datetime(df["window"], unit="s", utc=True)

    keys = ["src_ip"] + (["app"] if by_app else [])
    group_keys = keys + ["window"]

    g = df.groupby(group_keys, sort=False)

    # Agrégations simples — rapides
    agg = g.agg(
        n_attempts = ("result",     "size"),
        n_fail     = ("is_fail",    "sum"),
        n_success  = ("is_success", "sum"),
    ).reset_index()

    # nunique séparé pour éviter l'explosion mémoire
    for col, out_col in [("user", "n_users"), ("app", "n_apps"),
                          ("user_agent", "n_user_agents"), ("country", "n_countries")]:
        if col in df.columns and df[col].notna().any():
            agg[out_col] = g[col].nunique().values
        else:
            agg[out_col] = 1

    agg = agg.rename(columns={"window": "ts"})

    # Features dérivées
    win_minutes = pd.to_timedelta(window).total_seconds() / 60.0
    agg["fail_rate"]       = agg["n_fail"] / agg["n_attempts"].replace(0, np.nan)
    agg["attempts_per_min"] = agg["n_attempts"] / win_minutes
    agg["success_after_fail"] = (
        (agg["n_success"] > 0) & (agg["n_fail"] >= min_fail_for_success_after_fail)
    ).astype(int)

    # Entropie utilisateurs
    gcols = group_keys
    user_counts = (
        df.groupby(gcols + ["user"], sort=False)
          .size()
          .rename("cnt")
          .reset_index()
    )
    ent = (
        user_counts.groupby(gcols)["cnt"]
        .apply(_entropy_from_counts)
        .rename("user_entropy")
        .reset_index()
        .rename(columns={"window": "ts"})
    )

    out = agg.merge(ent, on=keys + ["ts"], how="left")
    out["user_entropy"] = out["user_entropy"].fillna(0.0)
    out = out.sort_values(keys + ["ts"]).reset_index(drop=True)
    return out
