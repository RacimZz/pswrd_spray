from __future__ import annotations

"""CLI SHAP: calcule et affiche les explications pour les alertes filtrées.

Usage:
    python -m src.cli_shap \\
        --model   data/processed/model_iforest.joblib \\
        --scored  data/processed/scored.csv \\
        --alerts  data/processed/alerts_deduped.csv \\
        --outdir  data/processed/shap
"""

import argparse
from pathlib import Path

import pandas as pd

from .explainer import explain_all_alerts
from .model import IFModel


def main() -> int:
    p = argparse.ArgumentParser(prog="spray-ai-shap")
    p.add_argument("--model",   required=True, help="Chemin vers model_iforest.joblib")
    p.add_argument("--scored",  required=True, help="Chemin vers scored.csv")
    p.add_argument("--alerts",  required=True, help="Chemin vers alerts_deduped.csv (post-filtrées)")
    p.add_argument("--outdir",  required=True, help="Dossier de sortie SHAP")
    p.add_argument("--top-n",   type=int, default=5, help="Top N features par alerte")
    p.add_argument("--max-alerts", type=int, default=20)
    args = p.parse_args()

    model  = IFModel.load(args.model)
    scored = pd.read_csv(args.scored)
    alerts = pd.read_csv(args.alerts)

    print(f"Modèle chargé | {len(scored)} fenêtres | {len(alerts)} alertes à expliquer")

    explained = explain_all_alerts(
        model=model,
        scored=scored,
        alerts=alerts,
        outdir=args.outdir,
        top_n_features=args.top_n,
        max_alerts=args.max_alerts,
    )

    out = Path(args.outdir)
    explained.to_csv(out / "alerts_explained.csv", index=False)
    print(f"\n✅ Résultats sauvegardés dans {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
