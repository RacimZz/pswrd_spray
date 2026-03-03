from __future__ import annotations

"""CLI pour appliquer le post-filtrage sur des alertes existantes.

Usage:
    python -m src.cli_postfilter --alerts data/processed/alerts.csv --outdir data/processed
"""

import argparse
from pathlib import Path

import pandas as pd

from .postfilter import PostFilterConfig, postfilter_alerts


def main() -> int:
    p = argparse.ArgumentParser(prog="spray-ai-postfilter")
    p.add_argument("--alerts", required=True, help="CSV d'alertes brutes")
    p.add_argument("--outdir", required=True)
    p.add_argument("--min-attempts", type=int, default=20)
    p.add_argument("--min-fail", type=int, default=10)
    p.add_argument("--min-fail-rate", type=float, default=0.30)
    p.add_argument("--min-score", type=float, default=0.60)
    p.add_argument("--min-users-spray", type=int, default=10)
    p.add_argument("--dedup-window", default="30min")
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    alerts = pd.read_csv(args.alerts)
    print(f"Alertes brutes: {len(alerts)}")

    cfg = PostFilterConfig(
        min_attempts=args.min_attempts,
        min_fail=args.min_fail,
        min_fail_rate=args.min_fail_rate,
        min_anomaly_score=args.min_score,
        min_users_for_spray=args.min_users_spray,
        dedup_window=args.dedup_window,
    )

    filtered, eliminated, deduped = postfilter_alerts(alerts, cfg)

    # Export
    filtered.to_csv(out / "alerts_filtered.csv", index=False)
    eliminated.to_csv(out / "alerts_eliminated.csv", index=False)
    deduped.to_csv(out / "alerts_deduped.csv", index=False)

    print(f"\n📊 Résultats post-filtrage:")
    print(f"   Avant filtrage   : {len(alerts)} alertes")
    print(f"   Après filtrage   : {len(filtered)} alertes")
    print(f"   Éliminées        : {len(eliminated)} (faux positifs probables)")
    print(f"   Après dédup      : {len(deduped)} alertes uniques")

    if not filtered.empty and "priority" in filtered.columns:
        print(f"\n🎯 Distribution des priorités:")
        for p_level, count in filtered["priority"].value_counts().items():
            print(f"   {p_level:10s}: {count}")

    print(f"\n🔴 Alertes finales (après dédup):")
    cols = [c for c in ["src_ip", "ts", "priority", "anomaly_score", "n_attempts",
                         "n_fail", "n_users", "fail_rate", "success_after_fail"] if c in deduped.columns]
    print(deduped[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
