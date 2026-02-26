from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURE_COLS = [
    "n_attempts",
    "n_fail",
    "n_success",
    "n_users",
    "n_apps",
    "n_user_agents",
    "n_countries",
    "fail_rate",
    "attempts_per_min",
    "success_after_fail",
    "user_entropy",
]


@dataclass
class IFModel:
    scaler: StandardScaler
    model: IsolationForest
    feature_cols: List[str]

    def save(self, path: str) -> None:
        joblib.dump({"scaler": self.scaler, "model": self.model, "feature_cols": self.feature_cols}, path)

    @staticmethod
    def load(path: str) -> "IFModel":
        obj = joblib.load(path)
        return IFModel(scaler=obj["scaler"], model=obj["model"], feature_cols=obj["feature_cols"])


def train_isolation_forest(
    features: pd.DataFrame,
    feature_cols: List[str] = DEFAULT_FEATURE_COLS,
    contamination: float = 0.01,
    n_estimators: int = 300,
    random_state: int = 42,
) -> IFModel:
    X = features[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(Xn)
    return IFModel(scaler=scaler, model=model, feature_cols=feature_cols)


def score_isolation_forest(m: IFModel, features: pd.DataFrame) -> pd.DataFrame:
    X = features[m.feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xn = m.scaler.transform(X)

    # score_samples: higher is more normal; we invert to have higher = more anomalous
    anomaly_score = -m.model.score_samples(Xn)
    is_anomaly = (m.model.predict(Xn) == -1).astype(int)

    out = features.copy()
    out["anomaly_score"] = anomaly_score
    out["is_anomaly"] = is_anomaly
    return out
