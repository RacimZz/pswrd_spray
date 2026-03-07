from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path

FEATURE_COLS = [
    "n_attempts", "n_fail", "n_success", "n_users", "n_apps",
    "n_user_agents", "n_countries", "fail_rate",
    "attempts_per_min", "success_after_fail", "user_entropy",
]

def train_and_save(scored_csv: str, label_col: str = "is_attack_ip", out_path: str = "models/xgb.joblib"):
    df = pd.read_csv(scored_csv)
    df = df.dropna(subset=[label_col])

    X = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df[label_col].astype(int)

    # Gestion du déséquilibre des classes
    ratio = int((y == 0).sum() / max((y == 1).sum(), 1))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=ratio,   # compense le déséquilibre
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
        ))
    ])

    # Validation croisée
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    print(f"F1 CV moyen : {scores.mean():.3f} (+/- {scores.std():.3f})")

    model.fit(X, y)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": FEATURE_COLS}, out_path)
    print(f"Modele sauvegarde : {out_path}")

if __name__ == "__main__":
    train_and_save("data/processed/rba/scored.csv", label_col="is_attack_window")
