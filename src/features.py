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
    """Compute features on non-overlapping fixed windows using resample.

    This avoids pandas groupby+rolling corner cases and gives exact distinct counts
    inside each window.

    Expected columns: ts (datetime UTC), user, src_ip, app, result.
    Optionals: reason, user_agent, country.
    """
    df = logs.copy()
    df["is_fail"] = (df["result"] == "fail").astype(int)
    df["is_success"] = (df["result"] == "success").astype(int)

    df = df.sort_values("ts").set_index("ts")

    keys = ["src_ip"] + (["app"] if by_app else [])

    agg = (
        df.groupby(keys)
          .resample(window)
          .agg(
              n_attempts=("result", "size"),
              n_fail=("is_fail", "sum"),
              n_success=("is_success", "sum"),
              n_users=("user", "nunique"),
              n_apps=("app", "nunique"),
              n_user_agents=("user_agent", "nunique"),
              n_countries=("country", "nunique"),
          )
          .reset_index()
    )

    # Derived
    agg["fail_rate"] = agg["n_fail"] / agg["n_attempts"].replace(0, np.nan)

    # attempts per minute (window duration)
    win_minutes = pd.to_timedelta(window).total_seconds() / 60.0
    agg["attempts_per_min"] = agg["n_attempts"] / win_minutes

    # simple triage
    agg["success_after_fail"] = ((agg["n_success"] > 0) & (agg["n_fail"] >= min_fail_for_success_after_fail)).astype(int)

    # Entropy of targeted users per window (per IP (and app))
    # We compute it from the raw events with an extra groupby; it's heavier but still fine for medium data.
    gcols = keys + [pd.Grouper(key="ts", freq=window)]
    user_counts = (
        df.reset_index()
          .groupby(gcols + ["user"]).size()
          .rename("cnt")
          .reset_index()
    )
    ent = (
        user_counts.groupby(gcols)["cnt"].apply(_entropy_from_counts)
        .rename("user_entropy")
        .reset_index()
        .rename(columns={"ts": "ts"})
    )

    out = agg.merge(ent, on=keys + ["ts"], how="left")
    out["user_entropy"] = out["user_entropy"].fillna(0.0)

    # Cleanup
    out = out.sort_values(keys + ["ts"]).reset_index(drop=True)
    return out
