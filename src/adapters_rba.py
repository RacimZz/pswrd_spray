from __future__ import annotations

"""Adapter RBA avec features enrichies (fenetres 1h + 24h + 7j + ASN suspicion)."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_rba(path: str, n_rows: Optional[int] = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fichier non trouve: {path}")

    df = pd.read_csv(path, nrows=n_rows)
    df["ts"]      = pd.to_datetime(df["Login Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    df["user"]    = df["User ID"].astype(str)
    df["src_ip"]  = df["IP Address"].astype(str)
    df["app"]     = "SSO"
    df["country"] = df["Country"].astype(str)
    df["rtt"]     = pd.to_numeric(df["Round-Trip Time [ms]"], errors="coerce")
    df["asn"]     = df["ASN"].astype(str)
    df["device"]  = df["Device Type"].astype(str)
    df["browser"] = df["Browser Name and Version"].astype(str)
    df["os"]      = df["OS Name and Version"].astype(str)
    df["ua"]      = df["User Agent String"].astype(str)
    df["result"]  = df["Login Successful"].astype(str).str.lower().map(
        {"true": "success", "false": "fail"}
    ).fillna("fail")
    df["reason"]      = None
    df["user_agent"]  = df["browser"]
    df["is_attack_ip"]        = df["Is Attack IP"].astype(str).str.lower().map({"true":1,"false":0}).fillna(0).astype(int)
    df["is_account_takeover"] = df["Is Account Takeover"].astype(str).str.lower().map({"true":1,"false":0}).fillna(0).astype(int)
    df["is_fail"]    = (df["result"] == "fail").astype(int)
    df["is_success"] = (df["result"] == "success").astype(int)

    cols = ["ts","user","src_ip","app","result","reason","user_agent","country",
            "rtt","asn","device","browser","os","ua",
            "is_fail","is_success","is_attack_ip","is_account_takeover"]
    return df[cols].sort_values("ts").reset_index(drop=True)


BOT_SIGNATURES = ["python","curl","requests","scrapy","bot","crawler","wget","go-http","java/","libwww"]


def _bot_flag(ua_series: pd.Series) -> pd.Series:
    lower = ua_series.str.lower().fillna("")
    return lower.apply(lambda s: int(any(sig in s for sig in BOT_SIGNATURES)))


def _entropy(counts: pd.Series) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = (counts / total).to_numpy(dtype=float)
    return float(-(p * np.log2(p + 1e-12)).sum())


def _agg_one_window(df_ts: pd.DataFrame, df_raw: pd.DataFrame, window: str, suffix: str) -> pd.DataFrame:
    base = (
        df_ts.groupby("src_ip")
             .resample(window)
             .agg(
                 n_attempts  = ("result",     "size"),
                 n_fail      = ("is_fail",    "sum"),
                 n_success   = ("is_success", "sum"),
                 n_users     = ("user",       "nunique"),
                 n_countries = ("country",    "nunique"),
                 n_asns      = ("asn",        "nunique"),
                 n_devices   = ("device",     "nunique"),
                 n_browsers  = ("browser",    "nunique"),
                 avg_rtt     = ("rtt",        "mean"),
                 rtt_std     = ("rtt",        "std"),
                 n_bot_ua    = ("is_bot_ua",  "sum"),
             )
             .reset_index()
    )
    base["fail_rate"]        = base["n_fail"] / base["n_attempts"].replace(0, np.nan)
    win_min                  = pd.to_timedelta(window).total_seconds() / 60.0
    base["attempts_per_min"] = base["n_attempts"] / win_min
    base["bot_ua_rate"]      = base["n_bot_ua"] / base["n_attempts"].replace(0, np.nan)
    base["rtt_is_robotic"]   = (base["rtt_std"].fillna(999) < 5).astype(int)
    base["success_after_fail"] = ((base["n_success"] > 0) & (base["n_fail"] >= 3)).astype(int)

    # user entropy
    uc = (
        df_raw.groupby(["src_ip", pd.Grouper(key="ts", freq=window), "user"])
              .size().rename("cnt").reset_index()
    )
    ent = (
        uc.groupby(["src_ip","ts"])["cnt"].apply(_entropy)
          .rename("user_entropy").reset_index()
    )
    base = base.merge(ent, on=["src_ip","ts"], how="left")
    base["user_entropy"] = base["user_entropy"].fillna(0.0)

    if suffix:
        rename = {c: f"{c}_{suffix}" for c in base.columns if c not in ["src_ip","ts"]}
        base = base.rename(columns=rename)
    return base


def _agg_7days(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Features de persistance sur 7 jours (par IP, fenetre glissante approximee par resample)."""
    df = df_raw.copy()
    df["date"] = df["ts"].dt.date

    per_ip_day = (
        df.groupby(["src_ip","date"])
          .agg(
              n_attempts_day = ("result",     "size"),
              n_fail_day     = ("is_fail",    "sum"),
              n_users_day    = ("user",       "nunique"),
              n_bot_ua_day   = ("is_bot_ua",  "sum"),
          )
          .reset_index()
    )

    # Agregation sur 7j glissants (par IP)
    records = []
    for ip, grp in per_ip_day.groupby("src_ip"):
        grp = grp.sort_values("date")
        n_active_days    = len(grp)
        n_attempts_7d    = int(grp["n_attempts_day"].sum())
        n_fail_7d        = int(grp["n_fail_day"].sum())
        n_users_7d       = int(grp["n_users_day"].max())
        bot_ua_rate_7d   = float(grp["n_bot_ua_day"].sum() / max(n_attempts_7d, 1))
        # Nombre de "pauses" entre jours consecutifs (signe d'un bot qui attend)
        dates_sorted = sorted(grp["date"].tolist())
        gaps = [(dates_sorted[i+1] - dates_sorted[i]).days for i in range(len(dates_sorted)-1)]
        n_gaps_over_1d   = int(sum(1 for g in gaps if g > 1))

        records.append({
            "src_ip":          ip,
            "n_active_days":   n_active_days,
            "n_attempts_7d":   n_attempts_7d,
            "n_fail_7d":       n_fail_7d,
            "n_users_7d":      n_users_7d,
            "bot_ua_rate_7d":  bot_ua_rate_7d,
            "n_gaps_over_1d":  n_gaps_over_1d,
        })

    return pd.DataFrame(records)


def _compute_asn_suspicion(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Score de suspicion par ASN: taux d'IPs d'attaque dans cet ASN."""
    asn_stats = (
        df_raw.groupby(["asn","src_ip"])["is_attack_ip"]
              .max().reset_index()
              .groupby("asn")["is_attack_ip"]
              .agg(asn_attack_rate=("mean"), asn_n_ips="count")
              .reset_index()
    )
    # Seulement si l'ASN a au moins 5 IPs connues (sinon bruit)
    asn_stats.loc[asn_stats["asn_n_ips"] < 5, "asn_attack_rate"] = 0.0
    return asn_stats[["asn","asn_attack_rate"]]


def compute_features_rba(
    df: pd.DataFrame,
    window_short: str = "1h",
    window_long: str  = "24h",
) -> pd.DataFrame:
    df = df.copy().sort_values("ts")
    df["is_bot_ua"] = _bot_flag(df["ua"])

    df_ts = df.set_index("ts")

    print("      Agregation fenetre courte...")
    feat_short = _agg_one_window(df_ts, df, window_short, "")

    print("      Agregation fenetre longue (24h)...")
    feat_long  = _agg_one_window(df_ts, df, window_long, "24h")

    print("      Features de persistance (7 jours)...")
    feat_7d = _agg_7days(df)

    print("      Score de suspicion ASN...")
    asn_susp = _compute_asn_suspicion(df)

    # Merge fenetres courte + longue
    feat_short["ts_day"] = feat_short["ts"].dt.floor(window_long)
    feat_long["ts_day"]  = feat_long["ts"].copy()
    feat = feat_short.merge(
        feat_long.drop(columns=["ts"]),
        on=["src_ip","ts_day"], how="left",
        suffixes=("","_dup"),
    )
    feat = feat.drop(columns=["ts_day"] + [c for c in feat.columns if c.endswith("_dup")])

    # Merge features 7j (une seule ligne par IP)
    feat = feat.merge(feat_7d, on="src_ip", how="left")

    # Merge ASN suspicion
    asn_per_ip = df.groupby("src_ip")["asn"].agg(lambda x: x.mode()[0] if len(x) > 0 else "unknown").reset_index()
    feat = feat.merge(asn_per_ip, on="src_ip", how="left")
    feat = feat.merge(asn_susp, on="asn", how="left")
    feat["asn_attack_rate"] = feat["asn_attack_rate"].fillna(0.0)
    feat = feat.drop(columns=["asn"], errors="ignore")

    # Fillna
    for col in feat.select_dtypes(include=[float, int]).columns:
        feat[col] = feat[col].fillna(0.0)

    return feat


RBA_FEATURE_COLS_UNSUPERVISED = [
    # Fenetre courte
    "n_attempts","n_fail","n_success","n_users",
    "fail_rate","attempts_per_min",
    "n_asns","n_devices","n_browsers","n_countries",
    "avg_rtt","rtt_std","rtt_is_robotic","bot_ua_rate",
    "success_after_fail","user_entropy",
    # Fenetre longue 24h
    "n_attempts_24h","n_fail_24h","n_users_24h",
    "fail_rate_24h","bot_ua_rate_24h","rtt_is_robotic_24h",
    # Persistance 7j
    "n_active_days","n_attempts_7d","n_fail_7d",
    "n_users_7d","bot_ua_rate_7d","n_gaps_over_1d",
    # ASN
    "asn_attack_rate",
]

RBA_FEATURE_COLS_SUPERVISED = RBA_FEATURE_COLS_UNSUPERVISED  # memes features, modele different


def evaluate_rba(scored: pd.DataFrame, labels: pd.DataFrame, window: str = "1h") -> dict:
    from sklearn.metrics import classification_report, confusion_matrix

    labels_agg = (
        labels.set_index("ts")
              .groupby("src_ip").resample(window)
              .agg(has_attack_ip=("is_attack_ip","max"), has_takeover=("is_account_takeover","max"))
              .reset_index()
    )
    merged = scored.merge(labels_agg, on=["src_ip","ts"], how="left")
    merged["has_attack_ip"] = merged["has_attack_ip"].fillna(0).astype(int)
    y_true = merged["has_attack_ip"].values
    y_pred = merged["is_anomaly"].values
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "vs_attack_ip":     report,
        "confusion_matrix": cm,
        "n_attack_ips":     int(merged["has_attack_ip"].sum()),
        "n_anomalies":      int(merged["is_anomaly"].sum()),
        "n_takeovers":      int(merged["has_takeover"].sum()),
    }
