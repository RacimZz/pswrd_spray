from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")

"""Explicabilité des alertes avec SHAP.

But: pour chaque alerte, dire POURQUOI le score d'anomalie est élevé,
en termes lisibles par un analyste SOC.

Fonctionnement:
    1) On entraîne un TreeExplainer SHAP sur le modèle IsolationForest
    2) Pour chaque alerte, on calcule les SHAP values (contribution de chaque feature)
    3) On génère un texte lisible + une figure par alerte

Référence:
    "Anomaly detection and Explanation with Isolation Forest and SHAP"
    Microsoft Sentinel blog, 2023
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Textes humains pour chaque feature (pour le rapport SOC)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_DESCRIPTIONS = {
    "n_attempts":          "Nombre total de tentatives de connexion",
    "n_fail":              "Nombre d'échecs de connexion",
    "n_success":           "Nombre de connexions réussies",
    "n_users":             "Nombre de comptes distincts ciblés",
    "n_apps":              "Nombre d'applications distinctes ciblées",
    "n_user_agents":       "Diversité des outils/navigateurs utilisés",
    "n_countries":         "Nombre de pays sources distincts",
    "fail_rate":           "Taux d'échec (n_fail / n_attempts)",
    "attempts_per_min":    "Rythme moyen de tentatives par minute",
    "success_after_fail":  "Succès détecté après de nombreux échecs",
    "user_entropy":        "Diversité statistique des comptes ciblés (entropie)",
}


def _check_shap():
    if not SHAP_AVAILABLE:
        raise ImportError(
            "SHAP n'est pas installé. Lance: pip install shap"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Calcul des SHAP values
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_values(
    model,        # IFModel (notre dataclass)
    features: pd.DataFrame,
) -> tuple:
    """Calcule les SHAP values pour toutes les fenêtres.

    Args:
        model: IFModel entraîné (contient scaler + IsolationForest + feature_cols)
        features: DataFrame complet des features (scored ou non)

    Returns:
        (shap_values_array, X_display)
        - shap_values_array : array (n_samples, n_features) de contributions SHAP
        - X_display         : DataFrame avec les valeurs originales (non standardisées)
    """
    _check_shap()

    X = features[model.feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Valeurs standardisées (ce que le modèle voit)
    Xn = model.scaler.transform(X)
    Xn_df = pd.DataFrame(Xn, columns=model.feature_cols)

    # TreeExplainer: adapté à IsolationForest (arbre de décision interne)
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(Xn_df)

    return shap_values, X


def explain_alert(
    alert_row: pd.Series,
    shap_row: np.ndarray,
    feature_cols: List[str],
    top_n: int = 5,
) -> dict:
    """Génère une explication lisible pour une alerte.

    Args:
        alert_row  : ligne du DataFrame d'alertes (1 alerte)
        shap_row   : SHAP values pour cette ligne (1D array)
        feature_cols: noms des features dans l'ordre
        top_n      : nombre de features à afficher

    Returns:
        dict avec:
            - summary    : texte court (1 ligne)
            - top_factors: liste des top features qui ont le plus contribué
            - text_report: texte multi-lignes pour le rapport
    """
    # Associe chaque feature à sa valeur SHAP
    contributions = pd.Series(shap_row, index=feature_cols)

    # Tri par valeur absolue (impact total, positif ou négatif)
    top = contributions.abs().nlargest(top_n)

    factors = []
    for feat in top.index:
        shap_val = float(contributions[feat])
        raw_val = float(alert_row.get(feat, np.nan))
        desc = FEATURE_DESCRIPTIONS.get(feat, feat)
        direction = "↑ HAUSSE le score" if shap_val > 0 else "↓ BAISSE le score"
        factors.append({
            "feature":     feat,
            "description": desc,
            "raw_value":   raw_val,
            "shap_value":  shap_val,
            "direction":   direction,
        })

    # Résumé 1 ligne
    top1 = factors[0]
    summary = (
        f"Alerte {alert_row.get('priority','?')} sur {alert_row.get('src_ip','?')} "
        f"[score={float(alert_row.get('anomaly_score',0)):.3f}] "
        f"— facteur principal: {top1['description']} = {top1['raw_value']:.1f}"
    )

    # Rapport texte complet
    lines = [
        f"=== EXPLICATION ALERTE ===",
        f"IP          : {alert_row.get('src_ip','?')}",
        f"Fenêtre     : {alert_row.get('ts','?')}",
        f"Priorité    : {alert_row.get('priority','?')}",
        f"Score       : {float(alert_row.get('anomaly_score',0)):.4f}",
        f"",
        f"Top {top_n} facteurs contributifs:",
    ]
    for i, f in enumerate(factors, 1):
        lines.append(
            f"  {i}. [{f['direction']}] {f['description']}"
            f" (valeur={f['raw_value']:.2f}, contribution SHAP={f['shap_value']:+.4f})"
        )

    return {
        "summary":     summary,
        "top_factors": factors,
        "text_report": "\n".join(lines),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figures SHAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_waterfall(
    alert_row: pd.Series,
    shap_row: np.ndarray,
    feature_cols: List[str],
    outpath: str,
    top_n: int = 8,
) -> None:
    """Génère un waterfall plot SHAP pour une alerte.

    Un waterfall montre comment chaque feature "pousse" le score
    vers le haut (rouge) ou vers le bas (bleu), depuis la valeur moyenne.
    """
    _check_shap()

    contributions = pd.Series(shap_row, index=feature_cols)
    top_feats = contributions.abs().nlargest(top_n).index.tolist()

    contrib_top = contributions[top_feats]
    values_top = pd.Series(
        [float(alert_row.get(f, 0)) for f in top_feats],
        index=top_feats
    )

    labels = [
        f"{f}\n(val={values_top[f]:.1f})"
        for f in top_feats
    ]

    colors = ["#e74c3c" if v > 0 else "#3498db" for v in contrib_top.values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels, contrib_top.values, color=colors)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Contribution SHAP (+ = hausse du score d'anomalie)")

    ip = alert_row.get("src_ip", "?")
    ts = str(alert_row.get("ts", "?"))[:16]
    score = float(alert_row.get("anomaly_score", 0))
    prio = alert_row.get("priority", "?")

    ax.set_title(f"[{prio}] {ip} | {ts} | score={score:.3f}", fontsize=10)
    plt.tight_layout()

    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_shap_summary(
    shap_values: np.ndarray,
    features: pd.DataFrame,
    feature_cols: List[str],
    outpath: str,
) -> None:
    """Génère un summary plot SHAP global (toutes les fenêtres)."""
    _check_shap()

    X = features[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fig, ax = plt.subplots(figsize=(8, 6))

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:10]

    ax.barh(
        [feature_cols[i] for i in order][::-1],
        mean_abs[order][::-1],
        color="#e74c3c"
    )
    ax.set_xlabel("Contribution SHAP moyenne (|valeur|)")
    ax.set_title("Features les plus importantes globalement")
    plt.tight_layout()

    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=160)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline complet
# ─────────────────────────────────────────────────────────────────────────────

def explain_all_alerts(
    model,
    scored: pd.DataFrame,
    alerts: pd.DataFrame,
    outdir: str,
    top_n_features: int = 5,
    max_alerts: int = 20,
) -> pd.DataFrame:
    """Calcule SHAP + explications pour toutes les alertes filtrées.

    Args:
        model   : IFModel entraîné
        scored  : DataFrame scoré complet (toutes les fenêtres)
        alerts  : DataFrame des alertes filtrées (post-filtrage)
        outdir  : dossier de sortie pour les figures
        top_n   : top N features à afficher par alerte
        max_alerts: nombre max d'alertes à expliquer

    Returns:
        DataFrame alerts enrichi avec colonnes:
            - shap_summary  : texte court
            - shap_report   : texte long (pour rapport)
            - shap_fig_path : chemin de la figure waterfall
    """
    _check_shap()

    out = pathlib.Path(outdir)
    (out / "shap_waterfalls").mkdir(parents=True, exist_ok=True)

    print("[SHAP] Calcul des SHAP values sur toutes les fenêtres...")
    shap_values, X_raw = compute_shap_values(model, scored)

    # Summary global
    print("[SHAP] Figure summary globale...")
    plot_shap_summary(shap_values, scored, model.feature_cols, str(out / "shap_summary.png"))

    # Associe chaque alerte à sa ligne dans scored
    # On joint sur src_ip + ts
    scored_idx = scored.reset_index(drop=True)
    scored_idx["_row_idx"] = scored_idx.index

    merge_cols = ["src_ip", "ts"]
    alerts_work = alerts.head(max_alerts).copy()

    explanations = []
    for i, alert_row in alerts_work.iterrows():
        # Trouve la ligne correspondante dans scored
        mask = (scored_idx["src_ip"] == alert_row["src_ip"]) & \
               (scored_idx["ts"].astype(str) == str(alert_row["ts"]))
        match = scored_idx.loc[mask]

        if match.empty:
            explanations.append({"shap_summary": "N/A", "shap_report": "N/A", "shap_fig_path": ""})
            continue

        row_idx = int(match["_row_idx"].iloc[0])
        shap_row = shap_values[row_idx]

        # Explication textuelle
        exp = explain_alert(alert_row, shap_row, model.feature_cols, top_n=top_n_features)

        # Figure waterfall
        ip_safe = str(alert_row.get("src_ip", "unknown")).replace(".", "_")
        ts_safe = str(alert_row.get("ts", ""))[:16].replace(":", "").replace(" ", "_").replace("+", "")
        fig_path = str(out / "shap_waterfalls" / f"alert_{ip_safe}_{ts_safe}.png")
        plot_shap_waterfall(alert_row, shap_row, model.feature_cols, fig_path, top_n=top_n_features)

        explanations.append({
            "shap_summary":  exp["summary"],
            "shap_report":   exp["text_report"],
            "shap_fig_path": fig_path,
        })

        print(f"    {exp['summary']}")

    exp_df = pd.DataFrame(explanations, index=alerts_work.index)
    alerts_explained = pd.concat([alerts_work, exp_df], axis=1)

    # Export texte
    report_path = out / "shap_text_reports.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        for _, row in alerts_explained.iterrows():
            f.write(row.get("shap_report", "") + "\n\n" + "─"*60 + "\n\n")

    print(f"\n[SHAP] Terminé. Figures: {out/'shap_waterfalls'}")
    print(f"[SHAP] Rapport texte: {report_path}")

    return alerts_explained

import pathlib as pathlib  # already imported above — just for the closure
