from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .alerts import build_alerts
from .features import compute_features_fixed_windows
from .model import DEFAULT_FEATURE_COLS, IFModel, score_isolation_forest, train_isolation_forest
from .reporting import save_basic_figures, write_latex_table
from .schema import normalize_logs, normalize_rba
from .settings import Settings


@dataclass
class RunResult:
    features_path: str
    scored_path: str
    alerts_path: str
    model_path: str


def run_pipeline(
    input_csv: str,
    outdir: str,
    window: str = "10min",
    by_app: bool = False,
    contamination: float = 0.01,
    n_estimators: int = 300,
    random_state: int = 42,
    make_figures: bool = True,
) -> RunResult:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_csv, low_memory=True)
    if "Login Timestamp" in raw.columns:
        logs = normalize_rba(raw)
    else:
        logs = normalize_logs(raw)


    feat = compute_features_fixed_windows(
        logs,
        window=window,
        by_app=by_app,
        min_fail_for_success_after_fail=Settings().min_fail_for_success_after_fail,
    )

    model = train_isolation_forest(
        feat,
        feature_cols=DEFAULT_FEATURE_COLS,
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
    )

    scored = score_isolation_forest(model, feat)

    keys = ["src_ip"] + (["app"] if by_app else [])
    alerts = build_alerts(scored, keys=keys, ts_col="ts")

    features_path = str(out / "features.csv")
    scored_path = str(out / "scored.csv")
    alerts_path = str(out / "alerts.csv")
    model_path = str(out / "model_iforest.joblib")

    feat.to_csv(features_path, index=False)
    scored.to_csv(scored_path, index=False)
    alerts.to_csv(alerts_path, index=False)
    model.save(model_path)

    if make_figures:
        figs_dir = out / "figures"
        save_basic_figures(scored, str(figs_dir), ts_col="ts")

        # write latex tables
        write_latex_table(alerts, str(out / "tables" / "alerts_top.tex"), max_rows=30)

    return RunResult(
        features_path=features_path,
        scored_path=scored_path,
        alerts_path=alerts_path,
        model_path=model_path,
    )
