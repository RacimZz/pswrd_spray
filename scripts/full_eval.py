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
    print(f"   {Path(f).name} → {len(df):,} lignes")

df_cic = pd.concat(dfs, ignore_index=True)
print(f"\n Total : {len(df_cic):,} flux")

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
print(" Modèle CIC-brut sauvegardé")

y_pred1 = (clf1.predict(X_cic_s) == -1).astype(int)
print(classification_report(y_cic, y_pred1, target_names=['BENIGN', 'ATTACK']))

for label in df_cic['Label'].unique():
    if label == 'BENIGN':
        continue
    idx = df_cic['Label'] == label
    r = y_pred1[idx].mean()
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

scaler2 = StandardScaler()
X_agg_ns = scaler2.fit_transform(X_agg[y_agg == 0])
X_agg_s  = scaler2.transform(X_agg)

clf2 = IsolationForest(contamination=0.01, n_estimators=100, random_state=42, n_jobs=-1)
clf2.fit(X_agg_ns)
joblib.dump({'model': clf2, 'scaler': scaler2, 'features': FEAT_AGG},
            OUT_DIR / "model_cic_agg.joblib")
print(" Modèle CIC-agrégé sauvegardé")

y_pred2 = (clf2.predict(X_agg_s) == -1).astype(int)
print(classification_report(y_agg, y_pred2, target_names=['BENIGN', 'ATTACK']))

for label in agg['true_label'].unique():
    if label == 'BENIGN':
        continue
    idx = agg['true_label'] == label
    r = y_pred2[idx].mean()
    results_summary.append({'dataset': 'CIC-agg', 'label': label, 'recall': round(r, 4), 'n': int(idx.sum())})
    print(f"  {label:40s} recall={r:.2%}  n={idx.sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# BLOC 3 — RBA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("BLOC 3 : RBA")
print("═"*60)

if RBA_CSV.exists():
    df_rba = pd.read_csv(RBA_CSV)
    df_rba.columns = df_rba.columns.str.strip()
    print(f"  {len(df_rba):,} lignes | colonnes : {df_rba.columns.tolist()[:8]}...")
    if 'is_attack' in df_rba.columns:
        X_rba = df_rba.drop(columns=['is_attack']).select_dtypes(include=[np.number])
        X_rba = X_rba.replace([np.inf, -np.inf], np.nan).fillna(0)
        y_rba = df_rba['is_attack']
        scaler3 = StandardScaler()
        X_rba_ns = scaler3.fit_transform(X_rba[y_rba == 0])
        X_rba_s  = scaler3.transform(X_rba)
        clf3 = IsolationForest(contamination=float(y_rba.mean()), n_estimators=100, random_state=42, n_jobs=-1)
        clf3.fit(X_rba_ns)
        joblib.dump({'model': clf3, 'scaler': scaler3, 'features': list(X_rba.columns)},
                    OUT_DIR / "model_rba.joblib")
        print("💾 Modèle RBA sauvegardé")
        y_pred3 = (clf3.predict(X_rba_s) == -1).astype(int)
        print(classification_report(y_rba, y_pred3, target_names=['BENIGN', 'ATTACK']))
        r = y_pred3[y_rba == 1].mean()
        results_summary.append({'dataset': 'RBA', 'label': 'attack', 'recall': round(r, 4), 'n': int(y_rba.sum())})
    else:
        print("  ⚠️  Colonne 'is_attack' non trouvée — adapte le script pour RBA")
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
    print(f"  {len(df_synth):,} lignes | colonnes : {df_synth.columns.tolist()[:8]}...")
    if 'is_attack' in df_synth.columns:
        X_s = df_synth.drop(columns=['is_attack']).select_dtypes(include=[np.number])
        X_s = X_s.replace([np.inf, -np.inf], np.nan).fillna(0)
        y_s = df_synth['is_attack']
        scaler4 = StandardScaler()
        X_sn = scaler4.fit_transform(X_s[y_s == 0])
        X_ss = scaler4.transform(X_s)
        clf4 = IsolationForest(contamination=float(y_s.mean()), n_estimators=100, random_state=42, n_jobs=-1)
        clf4.fit(X_sn)
        joblib.dump({'model': clf4, 'scaler': scaler4, 'features': list(X_s.columns)},
                    OUT_DIR / "model_synth.joblib")
        print(" Modèle Synthétique sauvegardé")
        y_pred4 = (clf4.predict(X_ss) == -1).astype(int)
        print(classification_report(y_s, y_pred4, target_names=['BENIGN', 'ATTACK']))
        r = y_pred4[y_s == 1].mean()
        results_summary.append({'dataset': 'Synthétique', 'label': 'attack', 'recall': round(r, 4), 'n': int(y_s.sum())})
    else:
        print("  ⚠️  Colonne 'is_attack' non trouvée — adapte le script pour synthétique")
else:
    print(f"  ⚠️  {SYNTH_CSV} non trouvé — bloc ignoré")

# ══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("RÉSUMÉ COMPLET")
print("═"*60)
df_summary = pd.DataFrame(results_summary)
print(df_summary.sort_values(['dataset', 'recall'], ascending=[True, False]).to_string(index=False))
df_summary.to_csv(OUT_DIR / "full_eval_summary.csv", index=False)
print(f"\n Résumé → {OUT_DIR}/full_eval_summary.csv")
print(" Modèles → model_cic_brut.joblib / model_cic_agg.joblib / model_rba.joblib / model_synth.joblib")
