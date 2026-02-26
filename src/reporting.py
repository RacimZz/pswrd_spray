from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_basic_figures(scored: pd.DataFrame, outdir: str, ts_col: str = "ts") -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Score distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(scored["anomaly_score"], bins=60)
    plt.title("Distribution des scores d'anomalie")
    plt.tight_layout()
    plt.savefig(out / "anomaly_score_hist.png", dpi=160)
    plt.close()

    # 2) Alerts over time
    tmp = scored.copy()
    tmp[ts_col] = pd.to_datetime(tmp[ts_col], utc=True)
    by_t = tmp.groupby(pd.Grouper(key=ts_col, freq="30min"))["is_anomaly"].sum().reset_index()

    plt.figure(figsize=(10, 4))
    plt.plot(by_t[ts_col], by_t["is_anomaly"])
    plt.title("Nombre d'anomalies par 30 minutes")
    plt.ylabel("# anomalies")
    plt.tight_layout()
    plt.savefig(out / "anomalies_over_time.png", dpi=160)
    plt.close()


def write_latex_table(df: pd.DataFrame, path: str, max_rows: int = 30) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sub = df.head(max_rows)
    latex = sub.to_latex(index=False)
    Path(path).write_text(latex, encoding="utf-8")
