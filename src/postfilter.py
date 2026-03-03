from __future__ import annotations

"""Post-filtrage des alertes brutes (IsolationForest).

But: réduire les faux positifs en appliquant des règles métier
après le score d'anomalie, avant de présenter les alertes à un analyste SOC.

Trois niveaux de règles:
    1) HARD FILTERS  : règles qui éliminent les alertes non-pertinentes
    2) TRIAGE RULES  : règles qui élèvent ou baissent la priorité d'une alerte
    3) DEDUP         : dédoublonnage (même IP qui alerte sur plusieurs fenêtres proches)
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PostFilterConfig:
    # Hard filters
    min_attempts: int = 20
    """Nombre minimum de tentatives dans la fenêtre pour garder une alerte.
    En dessous, le volume est trop faible pour être significatif."""

    min_fail: int = 10
    """Nombre minimum d'échecs. Une fenêtre avec 1 ou 2 échecs ne vaut pas une alerte."""

    min_fail_rate: float = 0.3
    """Taux d'échec minimum. En dessous de 30%, c'est probablement du bruit normal."""

    # Pour la règle spray spécifique
    min_users_for_spray: int = 10
    """Nombre minimum de users distincts pour classifier comme spray suspect."""

    # Triage
    critical_success_after_fail: bool = True
    """Si True, une alerte avec success_after_fail=1 devient automatiquement CRITICAL."""

    high_fail_rate_threshold: float = 0.85
    """fail_rate > seuil ET volume suffisant → HIGH."""

    high_users_threshold: int = 30
    """n_users > seuil → SPRAY suspect → HIGH ou CRITICAL."""

    # Déduplication
    dedup_window: str = "30min"
    """Regroupe les alertes de la même IP dans une fenêtre de temps pour en faire 1 seule."""

    # Score minimum final
    min_anomaly_score: float = 0.60
    """Filtre les alertes avec un score d'anomalie trop faible."""


# ─────────────────────────────────────────────────────────────────────────────
# Niveaux de priorité
# ─────────────────────────────────────────────────────────────────────────────

PRIORITY_CRITICAL = "CRITICAL"
PRIORITY_HIGH     = "HIGH"
PRIORITY_MEDIUM   = "MEDIUM"
PRIORITY_LOW      = "LOW"


def assign_priority(row: pd.Series, cfg: PostFilterConfig) -> str:
    """Assigne une priorité à une alerte selon des règles métier.

    Règles (par ordre de priorité décroissante):
        CRITICAL : succès détecté après beaucoup d'échecs (compromission possible)
                   OU spray massif (n_users très élevé + fail_rate élevé)
        HIGH     : fail_rate élevé + volume moyen OU beaucoup de users distincts
        MEDIUM   : volume anormal mais pas de signal fort de compromission
        LOW      : le reste
    """
    success_after_fail = int(row.get("success_after_fail", 0))
    n_users = int(row.get("n_users", 0))
    fail_rate = float(row.get("fail_rate", 0.0))
    n_attempts = int(row.get("n_attempts", 0))
    anomaly_score = float(row.get("anomaly_score", 0.0))

    # CRITICAL
    if cfg.critical_success_after_fail and success_after_fail == 1:
        return PRIORITY_CRITICAL
    if n_users >= cfg.high_users_threshold * 2 and fail_rate >= 0.90:
        return PRIORITY_CRITICAL

    # HIGH
    if n_users >= cfg.high_users_threshold and fail_rate >= cfg.high_fail_rate_threshold:
        return PRIORITY_HIGH
    if fail_rate >= cfg.high_fail_rate_threshold and n_attempts >= 100:
        return PRIORITY_HIGH

    # MEDIUM
    if anomaly_score >= 0.75 or n_users >= cfg.min_users_for_spray:
        return PRIORITY_MEDIUM

    return PRIORITY_LOW


# ─────────────────────────────────────────────────────────────────────────────
# Hard filters
# ─────────────────────────────────────────────────────────────────────────────

def apply_hard_filters(alerts: pd.DataFrame, cfg: PostFilterConfig) -> pd.DataFrame:
    """Applique les filtres stricts: élimine ce qui est clairement du bruit.

    Retourne un DataFrame filtré + une colonne 'filter_reason' pour les éliminés.
    """
    mask_keep = (
        (alerts["n_attempts"] >= cfg.min_attempts)
        & (alerts["n_fail"] >= cfg.min_fail)
        & (alerts["fail_rate"].fillna(0.0) >= cfg.min_fail_rate)
        & (alerts["anomaly_score"] >= cfg.min_anomaly_score)
    )

    eliminated = alerts.loc[~mask_keep].copy()
    kept = alerts.loc[mask_keep].copy()

    # Raison d'élimination (utile pour le debug + rapport)
    def filter_reason(r):
        reasons = []
        if r["n_attempts"] < cfg.min_attempts:
            reasons.append(f"n_attempts={int(r['n_attempts'])}<{cfg.min_attempts}")
        if r["n_fail"] < cfg.min_fail:
            reasons.append(f"n_fail={int(r['n_fail'])}<{cfg.min_fail}")
        if float(r.get("fail_rate", 0.0)) < cfg.min_fail_rate:
            reasons.append(f"fail_rate={float(r.get('fail_rate',0)):.2f}<{cfg.min_fail_rate}")
        if float(r.get("anomaly_score", 0.0)) < cfg.min_anomaly_score:
            reasons.append(f"score={float(r.get('anomaly_score',0)):.3f}<{cfg.min_anomaly_score}")
        return "; ".join(reasons)

    eliminated["filter_reason"] = eliminated.apply(filter_reason, axis=1)

    return kept, eliminated


# ─────────────────────────────────────────────────────────────────────────────
# Déduplication
# ─────────────────────────────────────────────────────────────────────────────

def dedup_alerts(alerts: pd.DataFrame, cfg: PostFilterConfig, ts_col: str = "ts") -> pd.DataFrame:
    """Regroupe les alertes de la même IP dans une fenêtre de temps.

    Pour une même IP qui alerte sur 5 fenêtres consécutives de 10min,
    on ne génère qu'une seule alerte (la pire = score max).
    """
    if alerts.empty:
        return alerts

    df = alerts.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.sort_values(["src_ip", ts_col])

    dedup_records = []
    for ip, group in df.groupby("src_ip"):
        group = group.set_index(ts_col).sort_index()

        # Rolling dedup : on regroupe par fenêtre cfg.dedup_window
        group_resampled = group.resample(cfg.dedup_window)

        for window_ts, window_data in group_resampled:
            if window_data.empty:
                continue

            # Garde la ligne avec le score le plus élevé
            best = window_data.loc[window_data["anomaly_score"].idxmax()].copy()
            best["n_windows_merged"] = len(window_data)
            best["ts_first"] = window_data.index.min()
            best["ts_last"] = window_data.index.max()
            best.name = window_ts
            dedup_records.append(best)

    if not dedup_records:
        return pd.DataFrame(columns=alerts.columns)

    deduped = pd.DataFrame(dedup_records)
    deduped.index.name = ts_col
    deduped = deduped.reset_index()

    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def postfilter_alerts(
    alerts: pd.DataFrame,
    cfg: Optional[PostFilterConfig] = None,
    ts_col: str = "ts",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pipeline complet de post-filtrage.

    Args:
        alerts: DataFrame brut d'alertes (is_anomaly=1)
        cfg: configuration du filtre
        ts_col: nom de la colonne timestamp

    Returns:
        (filtered, eliminated, deduped)
        - filtered  : alertes qui passent les filtres durs + priorité
        - eliminated: alertes éliminées (avec raison)
        - deduped   : alertes dédoublonnées (1 ligne par IP par 30min)
    """
    if cfg is None:
        cfg = PostFilterConfig()

    if alerts.empty:
        empty = alerts.copy()
        return empty, empty, empty

    # 1) Hard filters
    filtered, eliminated = apply_hard_filters(alerts, cfg)

    if filtered.empty:
        return filtered, eliminated, filtered.copy()

    # 2) Assignation des priorités
    filtered["priority"] = filtered.apply(lambda r: assign_priority(r, cfg), axis=1)

    # Ordre de priorité pour le tri
    priority_order = {PRIORITY_CRITICAL: 0, PRIORITY_HIGH: 1, PRIORITY_MEDIUM: 2, PRIORITY_LOW: 3}
    filtered["priority_rank"] = filtered["priority"].map(priority_order).fillna(99)
    filtered = filtered.sort_values(["priority_rank", "anomaly_score"], ascending=[True, False])
    filtered = filtered.drop(columns=["priority_rank"])

    # 3) Déduplication
    deduped = dedup_alerts(filtered, cfg, ts_col=ts_col)
    if "priority" in filtered.columns and "priority" not in deduped.columns:
        deduped["priority"] = PRIORITY_MEDIUM

    return filtered, eliminated, deduped
