from __future__ import annotations

from typing import List

import pandas as pd


def build_alerts(scored: pd.DataFrame, keys: List[str], ts_col: str = "ts") -> pd.DataFrame:
    """Select anomalies and format a SOC-friendly alerts table."""
    cols = keys + [ts_col, "anomaly_score", "n_attempts", "n_fail", "n_success", "n_users", "fail_rate", "attempts_per_min", "success_after_fail", "user_entropy"]
    cols = [c for c in cols if c in scored.columns]

    alerts = scored.loc[scored["is_anomaly"] == 1, cols].copy()
    alerts = alerts.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    # Human readable summary
    def summarize(r):
        return (
            f"score={r['anomaly_score']:.3f}, attempts={int(r['n_attempts'])}, fail={int(r['n_fail'])}, users={int(r['n_users'])}, "
            f"fail_rate={float(r['fail_rate']):.2f}, success_after_fail={int(r.get('success_after_fail', 0))}"
        )

    alerts["summary"] = alerts.apply(summarize, axis=1)
    return alerts
