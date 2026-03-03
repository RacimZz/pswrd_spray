from __future__ import annotations

"""Modele supervise XGBoost pour le dataset RBA.

On dispose des labels (Is Attack IP) -> on peut entrainer un modele supervise
bien plus performant que IsolationForest.

Etapes:
    1) Split train/test (70/30, stratifie)
    2) SMOTE pour rééquilibrer les classes (9.5% attaques vs 90.5% normal)
    3) Entrainement XGBoost
    4) Optimisation du seuil de decision (max recall >= 0.80)
    5) Evaluation et export
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


def _check_deps():
    missing = []
    try:
        import xgboost
    except ImportError:
        missing.append("xgboost")
    try:
        import imblearn
    except ImportError:
        missing.append("imbalanced-learn")
    if missing:
        raise ImportError(
            f"Packages manquants: {missing}\n"
            f"Lance: pip install {' '.join(missing)}"
        )


@dataclass
class SupervisedModel:
    model: object
    scaler: object
    feature_cols: List[str]
    threshold: float

    def save(self, path: str) -> None:
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
            "threshold": self.threshold,
        }, path)

    @staticmethod
    def load(path: str) -> "SupervisedModel":
        obj = joblib.load(path)
        return SupervisedModel(**obj)


def train_xgboost(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_cols: List[str],
    window: str = "1h",
    target_recall: float = 0.80,
    test_size: float = 0.30,
    random_state: int = 42,
) -> Tuple["SupervisedModel", dict]:
    _check_deps()

    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
    import xgboost as xgb

    # Aligner les labels sur les features (par src_ip + ts)
    labels_agg = (
        labels.set_index("ts")
              .groupby("src_ip").resample(window)
              .agg(y=("is_attack_ip","max"))
              .reset_index()
    )
    feat_labeled = features.merge(labels_agg, on=["src_ip","ts"], how="left")
    feat_labeled["y"] = feat_labeled["y"].fillna(0).astype(int)

    available = [c for c in feature_cols if c in feat_labeled.columns]
    X = feat_labeled[available].replace([np.inf,-np.inf], np.nan).fillna(0.0).values
    y = feat_labeled["y"].values

    print(f"      Dataset: {len(X)} exemples | {int(y.sum())} positifs ({100*y.mean():.1f}%)")

    # Split stratifie
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Standardisation
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # SMOTE sur le train uniquement
    print("      SMOTE reequilibrage...")
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_train_r, y_train_r = smote.fit_resample(X_train_s, y_train)
    print(f"      Apres SMOTE: {len(X_train_r)} exemples | {int(y_train_r.sum())} positifs")

    # XGBoost
    print("      Entrainement XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,   # SMOTE a deja rééquilibre
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train_r, y_train_r)

    # Optimisation du seuil
    print(f"      Optimisation du seuil (target recall >= {target_recall})...")
    proba_test = model.predict_proba(X_test_s)[:,1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba_test)

    # Cherche le seuil qui donne recall >= target avec la meilleure precision
    idx = np.where(recalls[:-1] >= target_recall)[0]
    if len(idx) == 0:
        print(f"      Recall {target_recall} non atteignable, on prend le max disponible")
        best_threshold = float(thresholds[np.argmax(recalls[:-1])])
    else:
        best_threshold = float(thresholds[idx[np.argmax(precisions[idx])]])

    print(f"      Seuil optimal: {best_threshold:.4f}")

    # Evaluation finale
    y_pred = (proba_test >= best_threshold).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "threshold":        best_threshold,
        "vs_attack_ip":     report,
        "confusion_matrix": cm,
        "n_test":           len(y_test),
        "n_positive_test":  int(y_test.sum()),
        "feature_importance": dict(zip(available, model.feature_importances_.tolist())),
    }

    sup_model = SupervisedModel(
        model=model,
        scaler=scaler,
        feature_cols=available,
        threshold=best_threshold,
    )
    return sup_model, metrics


def score_supervised(model: "SupervisedModel", features: pd.DataFrame) -> pd.DataFrame:
    X = features[model.feature_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0).values
    Xs = model.scaler.transform(X)
    proba = model.model.predict_proba(Xs)[:,1]
    is_anomaly = (proba >= model.threshold).astype(int)

    out = features.copy()
    out["anomaly_score"] = proba
    out["is_anomaly"]    = is_anomaly
    return out

# Alias export pour l'import dans cli_public.py
from .adapters_rba import RBA_FEATURE_COLS_UNSUPERVISED as RBA_FEATURE_COLS_SUPERVISED
