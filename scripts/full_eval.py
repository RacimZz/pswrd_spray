# scripts/full_eval.py
import glob
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── CHEMINS ──────────────────────────────────────────────────────────────────
BASE      = Path(r"C:\Users\racim\Desktop\ENSIMAG 1A\projet_perso\pswrd_spray")
CIC_DIR   = BASE / "data/public/CIC/"
RBA_CSV   = BASE / "data/public/rba/rba_1m.csv"
SYNTH_CSV = BASE / "data/raw/auth_logs.csv"
OUT_DIR   = BASE / "data/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEAT_AGG = ['n_flows', 'n_dst_ports', 'n_dst_ips', 'n_fail', 'fail_rate',
            'flows_per_min', 'unique_dst_per_min', 'port_entropy']

results_summary = []

# ══════════════════════════════════════════════════════════════════════════════
# BLOC 1 — CIC-IDS : flux brut
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("BLOC 1 : CIC-IDS — flux brut (sans agrégation)")
print("═"*60)

csv_files = glob.glob(str(CIC_DIR / "*.csv"))
dfs = []
for f in csv_files:
    df = pd.read_csv(f, low_memory=False)
    df.columns = df.columns.str.strip()
    dfs.append(df)
    print(f"  📄 {Path(f).name} → {len(df):,} lignes")

df_cic = pd.concat(dfs, ignore_index=True)
print(f"\n📊 Total : {len(df_cic):,} flux")

df_cic['is_attack'] = (df_cic['Label'] != 'BENIGN').astype(int)
X_cic = df_cic.drop(columns=['Label', 'is_attack']).select_dtypes(include=[np.number])
X_cic = X_cic.replace([np.inf, -np.inf], np.nan).fillna(0)
y_cic = df_cic['is_attack']

scaler1 = StandardScaler()
X_benign_s = scaler1.fit_transform(X_cic[y_cic == 0])
X_cic_s    = scaler1.transform(X_cic)

clf1 = IsolationForest(contamination=float(y_cic.mean()), n_estimators=100, random_state=42, n_jobs=-1)
clf1.fit(X_benign_s)
joblib.dump({'model': clf1, 'scaler': scaler1, 'features': list(X_cic.columns)},
            OUT_DIR / "model_cic_brut.joblib")
print("💾 Modèle CIC-brut sauvegardé")

y_pred1 = (clf1.predict(X_cic_s) == -1).astype(int)
print(classification_report(y_cic, y_pred1, target_names=['BENIGN', 'ATTACK']))

for label in df_cic['Label'].unique():
    if label == 'BENIGN':
        continue
    idx = df_cic['Label'] == label
    r = float(y_pred1[idx].mean())
    results_summary.append({'dataset': 'CIC-brut', 'label': label, 'recall': round(r, 4), 'n': int(idx.sum())})
    print(f"  {label:40s} recall={r:.2%}  n={idx.sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# BLOC 2 — CIC-IDS : agrégation par IP / 10min
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("BLOC 2 : CIC-IDS — agrégation par IP / 10min")
print("═"*60)

df_cic['timestamp'] = pd.date_range(start='2021-01-01', periods=len(df_cic), freq='s')
df_cic['is_fail']   = (df_cic['Label'] != 'BENIGN').astype(int)
df_cic['src_ip']    = 'IP_' + df_cic['Label'].str[:3] + (df_cic['Label'].apply(hash) % 1000).astype(str)
df_cic['dst_port']  = df_cic['Destination Port']
df_cic['window']    = df_cic['timestamp'].dt.floor('10min')

agg = df_cic.groupby(['src_ip', 'window']).agg(
    n_flows=('dst_port', 'count'),
    n_dst_ports=('dst_port', 'nunique'),
    n_dst_ips=('dst_port', 'nunique'),
    n_fail=('is_fail', 'sum'),
    fail_rate=('is_fail', 'mean'),
    true_label=('Label', lambda x: x[x != 'BENIGN'].iloc[0] if (x != 'BENIGN').any() else 'BENIGN')
).reset_index()

agg['flows_per_min']      = agg['n_flows'] * 6
agg['unique_dst_per_min'] = agg['n_dst_ips'] * 6
agg['port_entropy']       = np.log(agg['n_dst_ports'] + 1)

X_agg = agg[FEAT_AGG].fillna(0)
y_agg = (agg['true_label'] != 'BENIGN').astype(int)

contamination_agg = float(y_agg.mean())
print(f"  Contamination réelle : {contamination_agg:.2%}")

scaler2 = StandardScaler()
X_agg_ns = scaler2.fit_transform(X_agg[y_agg == 0])
X_agg_s  = scaler2.transform(X_agg)

clf2 = IsolationForest(contamination=contamination_agg, n_estimators=100, random_state=42, n_jobs=-1)
clf2.fit(X_agg_ns)
joblib.dump({'model': clf2, 'scaler': scaler2, 'features': FEAT_AGG},
            OUT_DIR / "model_cic_agg.joblib")
print("💾 Modèle CIC-agrégé sauvegardé")

y_pred2 = (clf2.predict(X_agg_s) == -1).astype(int)
print(classification_report(y_agg, y_pred2, target_names=['BENIGN', 'ATTACK']))

for label in agg['true_label'].unique():
    if label == 'BENIGN':
        continue
    idx = agg['true_label'] == label
    r = float(y_pred2[idx].mean())
    results_summary.append({'dataset': 'CIC-agg', 'label': label, 'recall': round(r, 4), 'n': int(idx.sum())})
    print(f"  {label:40s} recall={r:.2%}  n={idx.sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# BLOC 3 — RBA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("BLOC 3 : RBA")
print("═"*60)

if RBA_CSV.exists():
    df_rba = pd.read_csv(RBA_CSV, nrows=500_000)
    df_rba.columns = df_rba.columns.str.strip()

    df_rba = df_rba.rename(columns={
        'Login Timestamp': 'ts',
        'User ID':         'user',
        'IP Address':      'src_ip',
        'Country':         'country',
        'Login Successful':'login_success',
    })

    df_rba['ts']        = pd.to_datetime(df_rba['ts'], errors='coerce')
    df_rba              = df_rba.dropna(subset=['ts', 'src_ip'])
    df_rba['is_attack'] = df_rba['Is Attack IP'].fillna(0).astype(int)
    df_rba['is_fail']   = 1 - df_rba['login_success'].fillna(1).astype(int)
    df_rba['window']    = df_rba['ts'].dt.floor('10min')

    print(f"  Attaques dans le dataset : {df_rba['is_attack'].sum():,} / {len(df_rba):,} lignes")

    agg_rba = df_rba.groupby(['src_ip', 'window']).agg(
        n_flows=('src_ip', 'count'),
        n_users=('user', 'nunique'),
        n_countries=('country', 'nunique'),
        n_fail=('is_fail', 'sum'),
        fail_rate=('is_fail', 'mean'),
        is_attack_window=('is_attack', 'max')
    ).reset_index()

    agg_rba['flows_per_min']      = agg_rba['n_flows'] * 6
    agg_rba['unique_dst_per_min'] = agg_rba['n_users'] * 6
    agg_rba['port_entropy']       = np.log(agg_rba['n_countries'] + 1)

    FEAT_RBA = ['n_flows', 'n_users', 'n_countries', 'n_fail', 'fail_rate',
                'flows_per_min', 'unique_dst_per_min', 'port_entropy']

    X_rba = agg_rba[FEAT_RBA].fillna(0)
    y_rba = agg_rba['is_attack_window']

    print(f"  {len(agg_rba):,} fenêtres | fenêtres attaque : {y_rba.sum():,} ({y_rba.mean():.2%})")

    scaler3 = StandardScaler()
    X_rba_ns = scaler3.fit_transform(X_rba[y_rba == 0])
    X_rba_s  = scaler3.transform(X_rba)

    contamination_rba = max(float(y_rba.mean()), 0.001)
    clf3 = IsolationForest(contamination=contamination_rba, n_estimators=100, random_state=42, n_jobs=-1)
    clf3.fit(X_rba_ns)
    joblib.dump({'model': clf3, 'scaler': scaler3, 'features': FEAT_RBA},
                OUT_DIR / "model_rba.joblib")
    print("💾 Modèle RBA sauvegardé")

    y_pred3 = (clf3.predict(X_rba_s) == -1).astype(int)
    print(classification_report(y_rba, y_pred3, target_names=['BENIGN', 'ATTACK']))
    r = float(y_pred3[y_rba == 1].mean()) if y_rba.sum() > 0 else 0.0
    results_summary.append({'dataset': 'RBA', 'label': 'Is Attack IP', 'recall': round(r, 4), 'n': int(y_rba.sum())})
else:
    print(f"  ⚠️  {RBA_CSV} non trouvé — bloc ignoré")

# ══════════════════════════════════════════════════════════════════════════════
# BLOC 4 — Synthétique
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("BLOC 4 : Synthétique")
print("═"*60)

if SYNTH_CSV.exists():
    df_synth = pd.read_csv(SYNTH_CSV)
    df_synth.columns = df_synth.columns.str.strip()

    df_synth['ts']        = pd.to_datetime(df_synth['ts'], errors='coerce')
    df_synth              = df_synth.dropna(subset=['ts', 'src_ip'])
    df_synth['is_fail']   = (df_synth['result'] == 'fail').astype(int)
    df_synth['is_attack'] = df_synth['is_fail']
    df_synth['window']    = df_synth['ts'].dt.floor('10min')

    agg_synth = df_synth.groupby(['src_ip', 'window']).agg(
        n_flows=('src_ip', 'count'),
        n_users=('user', 'nunique'),
        n_apps=('app', 'nunique') if 'app' in df_synth.columns else ('src_ip', 'count'),
        n_fail=('is_fail', 'sum'),
        fail_rate=('is_fail', 'mean'),
        is_attack_window=('is_attack', 'max')
    ).reset_index()

    agg_synth['flows_per_min']      = agg_synth['n_flows'] * 6
    agg_synth['unique_dst_per_min'] = agg_synth['n_users'] * 6
    agg_synth['port_entropy']       = np.log(agg_synth['n_apps'] + 1)

    FEAT_SYNTH = ['n_flows', 'n_users', 'n_apps', 'n_fail', 'fail_rate',
                  'flows_per_min', 'unique_dst_per_min', 'port_entropy']

    X_synth = agg_synth[FEAT_SYNTH].fillna(0)
    y_synth = agg_synth['is_attack_window']

    print(f"  {len(agg_synth):,} fenêtres | attaques : {y_synth.mean():.2%}")

    scaler4 = StandardScaler()
    X_sn = scaler4.fit_transform(X_synth[y_synth == 0])
    X_ss = scaler4.transform(X_synth)

    clf4 = IsolationForest(contamination=max(float(y_synth.mean()), 0.001),
                           n_estimators=100, random_state=42, n_jobs=-1)
    clf4.fit(X_sn)
    joblib.dump({'model': clf4, 'scaler': scaler4, 'features': FEAT_SYNTH},
                OUT_DIR / "model_synth.joblib")
    print("💾 Modèle Synthétique sauvegardé")

    y_pred4 = (clf4.predict(X_ss) == -1).astype(int)
    print(classification_report(y_synth, y_pred4, target_names=['BENIGN', 'ATTACK']))
    r = float(y_pred4[y_synth == 1].mean()) if y_synth.sum() > 0 else 0.0
    results_summary.append({'dataset': 'Synthétique', 'label': 'attack', 'recall': round(r, 4), 'n': int(y_synth.sum())})
else:
    print(f"  ⚠️  {SYNTH_CSV} non trouvé — bloc ignoré")

# ══════════════════════════════════════════════════════════════════════════════
# BLOC 5 — UNSW-NB15
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("BLOC 5 : UNSW-NB15")
print("═"*60)

UNSW_DIR = BASE / "data/public/UNS"
train_path = UNSW_DIR / "UNSW_NB15_training-set.parquet"
test_path  = UNSW_DIR / "UNSW_NB15_testing-set.parquet"

if train_path.exists() and test_path.exists():
    df_train = pd.read_parquet(train_path)
    df_test  = pd.read_parquet(test_path)
    print(f"  Train : {len(df_train):,} lignes | Test : {len(df_test):,} lignes")

    # Features numériques uniquement
    drop_cols = ['attack_cat', 'label']
    X_train = df_train.drop(columns=drop_cols).select_dtypes(include=[np.number])
    y_train = df_train['label']

    X_test  = df_test.drop(columns=drop_cols).select_dtypes(include=[np.number])
    y_test  = df_test['label']

    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test  = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Aligne les colonnes train/test
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print(f"  Train attaques : {y_train.mean():.2%} | Test attaques : {y_test.mean():.2%}")

    scaler5 = StandardScaler()
    X_train_ns = scaler5.fit_transform(X_train[y_train == 0])
    X_test_s   = scaler5.transform(X_test)

    clf5 = IsolationForest(
        contamination=min(float(y_test.mean()), 0.499),
        n_estimators=100, random_state=42, n_jobs=-1
    )
    clf5.fit(X_train_ns)
    joblib.dump({'model': clf5, 'scaler': scaler5, 'features': list(X_train.columns)},
                OUT_DIR / "model_unsw.joblib")
    print("💾 Modèle UNSW-NB15 sauvegardé")

    y_pred5 = (clf5.predict(X_test_s) == -1).astype(int)
    print(classification_report(y_test, y_pred5, target_names=['BENIGN', 'ATTACK']))

    # Recall par catégorie
    df_test_copy = df_test.copy()
    df_test_copy['y_pred'] = y_pred5
    print("  Recall par catégorie :")
    for cat in df_test_copy['attack_cat'].unique():
        if cat == 'Normal':
            continue
        idx = df_test_copy['attack_cat'] == cat
        r = float(df_test_copy.loc[idx, 'y_pred'].mean())
        results_summary.append({'dataset': 'UNSW-NB15', 'label': cat, 'recall': round(r, 4), 'n': int(idx.sum())})
        print(f"    {cat:20s} recall={r:.2%}  n={idx.sum()}")
else:
    print(f"  ⚠️  Fichiers UNSW non trouvés dans {UNSW_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("RÉSUMÉ COMPLET")
print("═"*60)
df_summary = pd.DataFrame(results_summary)
print(df_summary.sort_values(['dataset', 'recall'], ascending=[True, False]).to_string(index=False))
df_summary.to_csv(OUT_DIR / "full_eval_summary.csv", index=False)
print(f"\n💾 Résumé → {OUT_DIR}/full_eval_summary.csv")
print("💾 Modèles → model_cic_brut.joblib / model_cic_agg.joblib / model_rba.joblib / model_synth.joblib")
