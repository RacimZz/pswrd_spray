from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class SynthConfig:
    start: str = "2026-02-01 08:00:00"
    minutes: int = 12 * 60
    n_users: int = 400
    n_normal_ips: int = 80
    apps: Tuple[str, ...] = ("SSO", "VPN", "MAIL")


def make_advanced_synth_logs(cfg: SynthConfig, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    start = pd.Timestamp(cfg.start, tz="UTC")
    users = [f"user{idx:04d}" for idx in range(cfg.n_users)]
    normal_ips = [f"198.51.100.{i}" for i in range(1, cfg.n_normal_ips + 1)]

    countries = ["FR", "DE", "ES", "GB", "US", "NL"]
    user_agents = ["Chrome", "Firefox", "Edge", "MobileSafari", "python-requests", "curl"]

    rows = []

    # Normal traffic
    for m in range(cfg.minutes):
        ts_base = start + pd.Timedelta(minutes=m)
        hour = ts_base.hour
        lam = 10 if 9 <= hour <= 18 else 4
        n_events = rng.poisson(lam)

        for _ in range(n_events):
            ts = ts_base + pd.Timedelta(seconds=int(rng.integers(0, 60)))
            user = rng.choice(users)
            ip = rng.choice(normal_ips)
            app = rng.choice(cfg.apps, p=[0.65, 0.2, 0.15])
            ua = rng.choice(user_agents, p=[0.35, 0.25, 0.2, 0.1, 0.07, 0.03])
            country = rng.choice(countries, p=[0.45, 0.2, 0.08, 0.1, 0.12, 0.05])

            p_fail = {"SSO": 0.06, "VPN": 0.10, "MAIL": 0.04}[app]
            fail = rng.random() < p_fail
            if fail:
                result = "fail"
                reason = rng.choice(["bad_password", "mfa_required", "user_typo"], p=[0.75, 0.05, 0.20])
            else:
                result = "success"
                reason = "ok"

            rows.append((ts, user, ip, app, result, reason, ua, country))

    # Scenario A: password spray
    spray_ip = "203.0.113.50"
    spray_start = 3 * 60 + 20
    spray_dur = 45
    spray_users = rng.choice(users, size=140, replace=False)

    for m in range(spray_start, spray_start + spray_dur):
        ts_base = start + pd.Timedelta(minutes=m)
        n_attempts = rng.poisson(18)
        for _ in range(n_attempts):
            ts = ts_base + pd.Timedelta(seconds=int(rng.integers(0, 60)))
            user = rng.choice(spray_users)
            app = rng.choice(cfg.apps, p=[0.85, 0.1, 0.05])
            ua = "python-requests"
            country = "US"
            success = rng.random() < 0.012
            result = "success" if success else "fail"
            reason = "ok" if success else "bad_password"
            rows.append((ts, user, spray_ip, app, result, reason, ua, country))

    # Scenario B: brute force
    brute_ip = "203.0.113.99"
    victim = rng.choice(users)
    brute_start = 6 * 60 + 10
    brute_dur = 20
    for m in range(brute_start, brute_start + brute_dur):
        ts_base = start + pd.Timedelta(minutes=m)
        n_attempts = rng.poisson(55)
        for _ in range(n_attempts):
            ts = ts_base + pd.Timedelta(seconds=int(rng.integers(0, 60)))
            app = rng.choice(cfg.apps, p=[0.7, 0.2, 0.1])
            ua = rng.choice(["curl", "python-requests"], p=[0.4, 0.6])
            country = "NL"
            rows.append((ts, victim, brute_ip, app, "fail", "bad_password", ua, country))

    # Scenario C: outage / misconfig
    outage_start = 9 * 60
    outage_dur = 25
    affected_users = rng.choice(users, size=60, replace=False)
    for m in range(outage_start, outage_start + outage_dur):
        ts_base = start + pd.Timedelta(minutes=m)
        n_events = rng.poisson(35)
        for _ in range(n_events):
            ts = ts_base + pd.Timedelta(seconds=int(rng.integers(0, 60)))
            user = rng.choice(affected_users)
            ip = rng.choice(normal_ips)
            app = "VPN"
            ua = rng.choice(["Chrome", "Edge", "MobileSafari"], p=[0.5, 0.3, 0.2])
            country = rng.choice(["FR", "DE", "GB"], p=[0.6, 0.25, 0.15])
            rows.append((ts, user, ip, app, "fail", "mfa_required", ua, country))

    df = pd.DataFrame(
        rows,
        columns=["ts", "user", "src_ip", "app", "result", "reason", "user_agent", "country"],
    )
    df = df.sort_values("ts").reset_index(drop=True)
    return df
