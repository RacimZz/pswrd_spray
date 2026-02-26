from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .pipeline import run_pipeline
from .synth import SynthConfig, make_advanced_synth_logs


def cmd_synth(args: argparse.Namespace) -> int:
    cfg = SynthConfig(minutes=args.minutes, n_users=args.n_users, n_normal_ips=args.n_normal_ips)
    logs = make_advanced_synth_logs(cfg, seed=args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    logs.to_csv(out, index=False)

    print(f"Wrote synthetic logs: {out} ({len(logs)} rows)")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    res = run_pipeline(
        input_csv=args.__dict__["in"],
        outdir=args.outdir,
        window=args.window,
        by_app=args.by_app,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        make_figures=not args.no_figures,
    )
    print("Pipeline done")
    print(res)
    return 0


def cmd_top(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.alerts)
    show = df.head(args.n)
    with pd.option_context("display.max_columns", 50, "display.width", 140):
        print(show)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spray-ai")
    sp = p.add_subparsers(dest="cmd", required=True)

    p_s = sp.add_parser("synth", help="Generate advanced synthetic auth logs")
    p_s.add_argument("--out", required=True, help="Output CSV path")
    p_s.add_argument("--minutes", type=int, default=12*60)
    p_s.add_argument("--n-users", type=int, default=400)
    p_s.add_argument("--n-normal-ips", type=int, default=80)
    p_s.add_argument("--seed", type=int, default=42)
    p_s.set_defaults(func=cmd_synth)

    p_r = sp.add_parser("run", help="Run full pipeline: features + model + scoring + alerts")
    p_r.add_argument("--in", required=True, dest="in")
    p_r.add_argument("--outdir", required=True)
    p_r.add_argument("--window", default="10min")
    p_r.add_argument("--by-app", action="store_true")
    p_r.add_argument("--contamination", type=float, default=0.01)
    p_r.add_argument("--n-estimators", type=int, default=300)
    p_r.add_argument("--random-state", type=int, default=42)
    p_r.add_argument("--no-figures", action="store_true")
    p_r.set_defaults(func=cmd_run)

    p_t = sp.add_parser("top", help="Show top alerts")
    p_t.add_argument("--alerts", required=True)
    p_t.add_argument("--n", type=int, default=20)
    p_t.set_defaults(func=cmd_top)

    return p


def main() -> int:
    p = build_parser()
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
