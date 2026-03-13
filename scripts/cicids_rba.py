# cicids_rba.py
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Features comme dans ton pipeline RBA
FEATURE_COLS = [
    "n_flows", "n_dst_ports", "n_dst_ips", "fail_rate",
    "flows_per_min", "unique_dst_per_min", "port_entropy"
]

DATA_DIR = r"C:\Users\racim\Desktop\ENSIMAG 1A\projet_perso\pswrd_spray\data\public\CIC"

# 1. CHARGE TOUS LES FICHIERS
csv_files = glob.glob(str(Path(DATA_DIR) / "**" / "*.csv"), recursive=True)
print(f" {len(csv_files)} fichiers CIC-IDS")

dfs = []
for f in csv_files:
    df = pd.read_csv(f, low_memory=False)
    df.columns = df.columns.str.strip()
    dfs.append(df)
    print(f"   {Path(f).name} → {len(df):,} lignes")

df_raw = pd.concat(dfs, ignore_index=True)
print(f"\n Total : {len(df_raw):,} flux réseau")

# 2. AJOUTE TIMESTAMP (manquant dans CIC-IDS CSV)
df_raw['timestamp'] = pd.date_range(
    start='2021-01-01', periods=len(df_raw), freq='S'
)

# 3. AGRÉGATION PAR IP SOURCE SUR 10 MIN
df_raw['src_ip'] = df_raw['Label'].map(lambda x: f"IP_{x[:3]}{hash(x) % 1000}")  # simule IP
df_raw['dst_ip'] = 'target_' + df_raw['Destination Port'].astype(str)
df_raw['dst_port'] = df_raw['Destination Port']
df_raw['is_fail'] = (df_raw['Label'] != 'BENIGN').astype(int)

# Fenêtres de 10min
df_raw = df_raw.sort_values('timestamp')
df_raw['window'] = df_raw['timestamp'].dt.floor('10min')

agg = df_raw.groupby(['src_ip', 'window']).agg({
    'dst_port': ['count', 'nunique'],
    'dst_ip': 'nunique',
    'is_fail': ['sum', 'mean'],
    'timestamp': ['count']  # nb flows
}).reset_index()

# Flatten les colonnes
agg.columns = ['src_ip', 'window', 'n_flows', 'n_dst_ports', 'n_dst_ips', 'n_fail', 'fail_rate', 'n_flows2']
agg = agg[['src_ip', 'window', 'n_flows', 'n_dst_ports', 'n_dst_ips', 'n_fail', 'fail_rate']]

# Features comportementales
agg['flows_per_min'] = agg['n_flows'] * 6  # 10min -> per min
agg['unique_dst_per_min'] = agg['n_dst_ips'] * 6
agg['port_entropy'] = np.log(agg['n_dst_ports'] + 1)

print(f"\n {len(agg):,} fenêtres agrégées")
print(f"  Attaques : {agg['fail_rate'].mean():.1%}")

# 4. TRAIN ISOLATION FOREST (sur normal seulement)
X = agg[FEATURE_COLS].fillna(0)
X_normal = X[agg['fail_rate'] == 0]

scaler = StandardScaler()
X_normal_scaled = scaler.fit_transform(X_normal)
X_scaled = scaler.transform(X)

model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X_normal_scaled)

# 5. PRÉDICTIONS
preds = model.predict(X_scaled)
scores = -model.score_samples(X_scaled)  # plus haut = plus suspect

agg['anomaly_score'] = scores
agg['is_anomaly'] = (preds == -1).astype(int)

# 6. RÉSULTATS
print("\n RÉSULTATS")
print(agg.groupby('fail_rate')['is_anomaly'].mean())

# Top alertes
alerts = agg[agg['is_anomaly'] == 1].sort_values('anomaly_score', ascending=False)
print(f"\n {len(alerts):,} alertes")
print(alerts[['src_ip', 'n_flows', 'n_dst_ips', 'fail_rate', 'anomaly_score']].head(10))

alerts.to_csv('cicids_rba_alerts.csv', index=False)
print("\n cicids_rba_alerts.csv généré")

# Recall par label réel
df_raw['window'] = df_raw['timestamp'].dt.floor('10min')
df_raw['src_ip_key'] = df_raw['src_ip']

label_per_window = df_raw.groupby(['src_ip_key', 'window'])['Label'].agg(
    lambda x: x[x != 'BENIGN'].iloc[0] if (x != 'BENIGN').any() else 'BENIGN'
).reset_index()
label_per_window.columns = ['src_ip', 'window', 'true_label']

agg = agg.merge(label_per_window, on=['src_ip', 'window'], how='left')

print("\n=== RECALL PAR TYPE D'ATTAQUE (avec agrégation) ===")
for label in agg['true_label'].dropna().unique():
    if label == 'BENIGN':
        continue
    subset = agg[agg['true_label'] == label]
    recall = subset['is_anomaly'].mean()
    print(f"  {label:40s} → recall = {recall:.2%}  ({len(subset)} fenêtres)")
