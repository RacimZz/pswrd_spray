# cicids_pipeline.py
import pandas as pd
import numpy as np
import glob
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ─── 1. MERGE DES 8 FICHIERS ───────────────────────────────────────────────
DATA_DIR = r"C:\Users\racim\Desktop\ENSIMAG 1A\projet_perso\pswrd_spray\data\public\CIC"

csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
print(f"Fichiers trouvés : {len(csv_files)}")

dfs = []
for f in csv_files:
    df = pd.read_csv(f, encoding='utf-8', low_memory=False)
    df.columns = df.columns.str.strip()  # enlève les espaces dans les noms
    dfs.append(df)
    print(f"  {os.path.basename(f)} → {len(df)} lignes")

df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal : {len(df)} lignes")
print(f"Labels : {df['Label'].value_counts().to_dict()}")

# ─── 2. NETTOYAGE ──────────────────────────────────────────────────────────
# Binarisation du label
df['is_attack'] = (df['Label'] != 'BENIGN').astype(int)

# Supprime colonnes non-numériques
df = df.drop(columns=['Label'])

# Remplace inf et NaN
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)

print(f"\nDistribution : {df['is_attack'].value_counts().to_dict()}")

# ─── 3. FEATURES / LABELS ──────────────────────────────────────────────────
X = df.drop(columns=['is_attack'])
y = df['is_attack']

# Garde uniquement colonnes numériques
X = X.select_dtypes(include=[np.number])

# ─── 4. TRAIN / TEST SPLIT ─────────────────────────────────────────────────
# Entraîne UNIQUEMENT sur le trafic BENIGN (unsupervised)
X_benign = X[y == 0]
X_test = X
y_test = y

print(f"\nTrain (BENIGN only) : {len(X_benign)} lignes")
print(f"Test (tout) : {len(X_test)} lignes")

# ─── 5. NORMALISATION ──────────────────────────────────────────────────────
scaler = StandardScaler()
X_benign_scaled = scaler.fit_transform(X_benign)
X_test_scaled = scaler.transform(X_test)

# ─── 6. ISOLATION FOREST ───────────────────────────────────────────────────
contamination = y.mean()  # ratio d'attaques réel
print(f"\nContamination estimée : {contamination:.4f}")

clf = IsolationForest(
    n_estimators=100,
    contamination=contamination,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_benign_scaled)

# ─── 7. PRÉDICTIONS ────────────────────────────────────────────────────────
# IsolationForest retourne -1 (anomalie) et 1 (normal)
# On convertit en 0/1
preds_raw = clf.predict(X_test_scaled)
y_pred = (preds_raw == -1).astype(int)

# ─── 8. RÉSULTATS ──────────────────────────────────────────────────────────
print("\n=== RÉSULTATS ===")
print(classification_report(y_test, y_pred, target_names=['BENIGN', 'ATTACK']))

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:\n{cm}")

# Sauvegarde des prédictions
df['y_pred'] = y_pred
df['anomaly_score'] = clf.score_samples(X_test_scaled)
df['is_attack'] = y_test.values
df[['is_attack', 'y_pred', 'anomaly_score']].to_csv('cicids_results.csv', index=False)
print("\nRésultats sauvegardés dans cicids_results.csv")

# Recall par type d'attaque
df_results = pd.read_csv('cicids_results.csv')

# Recharge les labels originaux
labels_all = []
for f in csv_files:
    tmp = pd.read_csv(f, encoding='utf-8', low_memory=False)
    tmp.columns = tmp.columns.str.strip()
    labels_all.extend(tmp['Label'].tolist())

df_results['Label'] = labels_all

# Recall par classe
print("\n=== RECALL PAR TYPE D'ATTAQUE ===")
for label in df_results['Label'].unique():
    if label == 'BENIGN':
        continue
    subset = df_results[df_results['Label'] == label]
    recall = subset['y_pred'].mean()
    print(f"  {label:40s} → recall = {recall:.2%}  ({len(subset)} échantillons)")
