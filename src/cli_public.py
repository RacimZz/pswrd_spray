from __future__ import annotations

"""CLI datasets publics (RBA avec mode supervise/non-supervise, LANL)."""

import argparse
import json
from pathlib import Path

import pandas as pd


def cmd_rba(args: argparse.Namespace) -> int:
    from .adapters_rba import (load_rba, compute_features_rba, evaluate_rba,
                                RBA_FEATURE_COLS_UNSUPERVISED, RBA_FEATURE_COLS_SUPERVISED)
    from .alerts import build_alerts
    from .reporting import save_basic_figures, write_latex_table

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] Chargement RBA ({args.rows} lignes max)...")
    df = load_rba(args.file, n_rows=args.rows)
    labels = df[["ts","src_ip","user","is_attack_ip","is_account_takeover"]].copy()
    print(f"      {len(df)} lignes | {int(df['is_attack_ip'].sum())} attack IPs | {int(df['is_account_takeover'].sum())} takeovers")
    df.to_csv(out / "logs_normalized.csv", index=False)
    labels.to_csv(out / "labels.csv", index=False)

    print(f"[2/7] Features enrichies (fenetres {args.window_short} + {args.window_long} + 7j + ASN)...")
    feat = compute_features_rba(df, window_short=args.window_short, window_long=args.window_long)
    feat.to_csv(out / "features.csv", index=False)
    print(f"      {len(feat)} fenetres | {feat['src_ip'].nunique()} IPs")

    # ── MODE SUPERVISE (XGBoost + SMOTE) ──────────────────────────────────────
    if args.supervised:
        from .supervised import train_xgboost, score_supervised, RBA_FEATURE_COLS_SUPERVISED as SF
        available = [c for c in RBA_FEATURE_COLS_SUPERVISED if c in feat.columns]

        print(f"[3/7] Entrainement XGBoost supervise (target recall={args.target_recall})...")
        sup_model, train_metrics = train_xgboost(
            feat, labels,
            feature_cols=available,
            window=args.window_short,
            target_recall=args.target_recall,
        )
        sup_model.save(str(out / "model_xgboost.joblib"))
        (out / "train_metrics.json").write_text(json.dumps(train_metrics, indent=2), encoding="utf-8")

        print("[4/7] Scoring supervise...")
        scored = score_supervised(sup_model, feat)

        # Importance des features
        fi = sorted(train_metrics["feature_importance"].items(), key=lambda x: x[1], reverse=True)
        print("\n      Top 10 features importantes:")
        for feat_name, imp in fi[:10]:
            print(f"        {feat_name:30s}: {imp:.4f}")

    # ── MODE NON SUPERVISE (IsolationForest) ──────────────────────────────────
    else:
        from .model import train_isolation_forest, score_isolation_forest
        available = [c for c in RBA_FEATURE_COLS_UNSUPERVISED if c in feat.columns]

        print(f"[3/7] Modele IsolationForest (contamination={args.contamination})...")
        model = train_isolation_forest(feat, feature_cols=available, contamination=args.contamination)
        model.save(str(out / "model_iforest.joblib"))

        print("[4/7] Scoring...")
        scored = score_isolation_forest(model, feat)

    scored.to_csv(out / "scored.csv", index=False)

    print("[5/7] Alertes...")
    alerts = build_alerts(scored, keys=["src_ip"], ts_col="ts")
    alerts.to_csv(out / "alerts.csv", index=False)

    print("[6/7] Evaluation vs labels reels...")
    from .adapters_rba import evaluate_rba
    metrics = evaluate_rba(scored, labels, window=args.window_short)
    (out / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("[7/7] Figures...")
    save_basic_figures(scored, str(out / "figures"))

    # ── Affichage final ───────────────────────────────────────────────────────
    print(f"\nRBA termine ({'supervise XGBoost' if args.supervised else 'non-supervise IForest'}).")
    print(f"   Anomalies detectees : {metrics['n_anomalies']}")
    print(f"   Attack IPs reelles  : {metrics['n_attack_ips']}")
    rpt = metrics["vs_attack_ip"]
    for cls, vals in rpt.items():
        if isinstance(vals, dict):
            print(f"   class {cls:12s}: precision={vals.get('precision',0):.3f}  recall={vals.get('recall',0):.3f}  f1={vals.get('f1-score',0):.3f}")

    print(f"\n   Top 10 alertes:")
    cols = [c for c in ["src_ip","ts","anomaly_score","n_attempts","n_fail",
                        "n_users","fail_rate","rtt_is_robotic","bot_ua_rate",
                        "n_active_days","asn_attack_rate"] if c in alerts.columns]
    print(alerts.head(10)[cols].to_string())
    return 0


def cmd_lanl(args: argparse.Namespace) -> int:
    from .adapters_lanl import load_lanl_chunk, compute_features_lanl
    from .model import train_isolation_forest, score_isolation_forest, DEFAULT_FEATURE_COLS
    from .alerts import build_alerts
    from .reporting import save_basic_figures, write_latex_table

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Chargement LANL ({args.rows} lignes max)...")
    logs = load_lanl_chunk(args.file, n_rows=args.rows, compressed=args.file.endswith(".bz2"))
    print(f"      {len(logs)} lignes chargees.")
    logs.to_csv(out / "logs_normalized.csv", index=False)

    print("[2/5] Features...")
    feat_by_computer, feat_by_user = compute_features_lanl(logs, window=args.window)
    feat_by_computer.to_csv(out / "features_by_computer.csv", index=False)

    print("[3/5] Modele...")
    model = train_isolation_forest(feat_by_computer, feature_cols=DEFAULT_FEATURE_COLS, contamination=args.contamination)

    print("[4/5] Scoring + alertes...")
    scored = score_isolation_forest(model, feat_by_computer)
    alerts = build_alerts(scored, keys=["src_ip"], ts_col="ts")
    scored.to_csv(out / "scored.csv", index=False)
    alerts.to_csv(out / "alerts.csv", index=False)
    model.save(str(out / "model.joblib"))

    print("[5/5] Figures...")
    save_basic_figures(scored, str(out / "figures"))
    write_latex_table(alerts, str(out / "tables" / "alerts_top.tex"), max_rows=30)
    print(f"\nLANL termine. Alertes: {len(alerts)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spray-ai-public")
    sp = p.add_subparsers(dest="cmd", required=True)

    p_r = sp.add_parser("rba")
    p_r.add_argument("--file",          required=True)
    p_r.add_argument("--rows",          type=int,   default=500_000)
    p_r.add_argument("--window-short",  default="1h")
    p_r.add_argument("--window-long",   default="24h")
    p_r.add_argument("--contamination", type=float, default=0.05)
    p_r.add_argument("--n-estimators",  type=int,   default=300)
    p_r.add_argument("--supervised",    action="store_true", help="Utiliser XGBoost supervise")
    p_r.add_argument("--target-recall", type=float, default=0.80, help="Recall cible (mode supervise)")
    p_r.add_argument("--outdir",        default="data/processed/rba")
    p_r.set_defaults(func=cmd_rba)

    p_l = sp.add_parser("lanl")
    p_l.add_argument("--file",          required=True)
    p_l.add_argument("--rows",          type=int,   default=5_000_000)
    p_l.add_argument("--window",        default="10min")
    p_l.add_argument("--contamination", type=float, default=0.005)
    p_l.add_argument("--outdir",        default="data/processed/lanl")
    p_l.set_defaults(func=cmd_lanl)

    return p


def main() -> int:
    p = build_parser()
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
