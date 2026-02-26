from __future__ import annotations

"""CLI pour les datasets publics (LANL, RBA).

Usage:
    # LANL (détection de mouvement latéral / credential hopping)
    python -m src.cli_public lanl --file data/public/lanl/lanl-auth-dataset-1-00.bz2 --rows 5000000 --outdir data/processed/lanl

    # RBA (détection password spray avec labels réels)
    python -m src.cli_public rba --file data/public/rba/rba.csv --rows 2000000 --outdir data/processed/rba
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def cmd_lanl(args: argparse.Namespace) -> int:
    from .adapters_lanl import load_lanl_chunk, compute_features_lanl
    from .model import train_isolation_forest, score_isolation_forest, DEFAULT_FEATURE_COLS
    from .alerts import build_alerts
    from .reporting import save_basic_figures, write_latex_table

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Chargement LANL ({args.rows} lignes max)...")
    logs = load_lanl_chunk(args.file, n_rows=args.rows, compressed=args.file.endswith(".bz2"))
    print(f"      {len(logs)} lignes chargées.")
    logs.to_csv(out / "logs_normalized.csv", index=False)

    print("[2/5] Calcul des features (by_computer + by_user)...")
    feat_by_computer, feat_by_user = compute_features_lanl(logs, window=args.window)
    feat_by_computer.to_csv(out / "features_by_computer.csv", index=False)
    feat_by_user.to_csv(out / "features_by_user.csv", index=False)
    print(f"      {len(feat_by_computer)} fenêtres (by_computer), {len(feat_by_user)} (by_user)")

    print("[3/5] Entraînement modèle (by_computer)...")
    model = train_isolation_forest(feat_by_computer, feature_cols=DEFAULT_FEATURE_COLS, contamination=args.contamination)

    print("[4/5] Scoring + export alertes...")
    scored = score_isolation_forest(model, feat_by_computer)
    alerts = build_alerts(scored, keys=["src_ip"], ts_col="ts")
    scored.to_csv(out / "scored.csv", index=False)
    alerts.to_csv(out / "alerts.csv", index=False)
    model.save(str(out / "model.joblib"))

    print("[5/5] Figures + tables LaTeX...")
    save_basic_figures(scored, str(out / "figures"))
    write_latex_table(alerts, str(out / "tables" / "alerts_top.tex"), max_rows=30)

    print(f"\n✅ LANL terminé. Alertes: {len(alerts)} | Top 10:")
    print(alerts.head(10)[["src_ip", "ts", "anomaly_score", "n_attempts", "n_users"]].to_string())
    return 0


def cmd_rba(args: argparse.Namespace) -> int:
    from .adapters_rba import load_rba, evaluate_rba
    from .features import compute_features_fixed_windows
    from .model import train_isolation_forest, score_isolation_forest, DEFAULT_FEATURE_COLS
    from .alerts import build_alerts
    from .reporting import save_basic_figures, write_latex_table

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Chargement RBA ({args.rows} lignes max)...")
    df = load_rba(args.file, n_rows=args.rows)
    labels = df[["ts", "src_ip", "user", "is_attack_ip", "is_account_takeover"]].copy()
    logs = df.drop(columns=["is_attack_ip", "is_account_takeover"])
    print(f"      {len(logs)} lignes | {int(df['is_attack_ip'].sum())} attack IPs | {int(df['is_account_takeover'].sum())} takeovers")
    logs.to_csv(out / "logs_normalized.csv", index=False)
    labels.to_csv(out / "labels.csv", index=False)

    print("[2/6] Features...")
    feat = compute_features_fixed_windows(logs, window=args.window)
    feat.to_csv(out / "features.csv", index=False)

    print("[3/6] Modèle...")
    model = train_isolation_forest(feat, feature_cols=DEFAULT_FEATURE_COLS, contamination=args.contamination)

    print("[4/6] Scoring...")
    scored = score_isolation_forest(model, feat)
    scored.to_csv(out / "scored.csv", index=False)

    print("[5/6] Alertes...")
    alerts = build_alerts(scored, keys=["src_ip"], ts_col="ts")
    alerts.to_csv(out / "alerts.csv", index=False)
    model.save(str(out / "model.joblib"))

    print("[6/6] Évaluation vs labels réels...")
    metrics = evaluate_rba(scored, labels)
    metrics_path = out / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ RBA terminé.")
    print(f"   Anomalies détectées: {metrics['n_anomalies']}")
    print(f"   Attack IPs dans le dataset: {metrics['n_attack_ips']}")
    print(f"   Rapport de classification (vs attack_ip label):")
    rpt = metrics["vs_attack_ip"]
    for cls, vals in rpt.items():
        if isinstance(vals, dict):
            print(f"     class {cls}: precision={vals.get('precision',0):.3f} recall={vals.get('recall',0):.3f} f1={vals.get('f1-score',0):.3f}")
    print(f"\n   Top 10 alertes:")
    print(alerts.head(10)[["src_ip", "ts", "anomaly_score", "n_attempts", "n_fail", "n_users"]].to_string())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spray-ai-public")
    sp = p.add_subparsers(dest="cmd", required=True)

    # LANL
    p_l = sp.add_parser("lanl", help="Run pipeline on LANL Auth dataset")
    p_l.add_argument("--file", required=True, help="Chemin vers le fichier .bz2 ou .txt")
    p_l.add_argument("--rows", type=int, default=5_000_000, help="Nb max de lignes à charger")
    p_l.add_argument("--window", default="10min")
    p_l.add_argument("--contamination", type=float, default=0.005)
    p_l.add_argument("--outdir", default="data/processed/lanl")
    p_l.set_defaults(func=cmd_lanl)

    # RBA
    p_r = sp.add_parser("rba", help="Run pipeline on RBA dataset (avec labels)")
    p_r.add_argument("--file", required=True, help="Chemin vers le CSV principal du RBA dataset")
    p_r.add_argument("--rows", type=int, default=2_000_000)
    p_r.add_argument("--window", default="10min")
    p_r.add_argument("--contamination", type=float, default=0.01)
    p_r.add_argument("--outdir", default="data/processed/rba")
    p_r.set_defaults(func=cmd_rba)

    return p


def main() -> int:
    p = build_parser()
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
