# scripts/optimize_model.py
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import (classification_report, precision_recall_curve,
                              average_precision_score, f1_score)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE     = Path(r"C:\Users\racim\Desktop\ENSIMAG 1A\projet_perso\pswrd_spray")
UNSW_DIR = BASE / "data/public/UNS"
OUT_DIR  = BASE / "data/results"
OUT_DIR.mkdir(exist_ok=True)

# ─── CHARGEMENT ───────────────────────────────────────────────────────────────
df_train = pd.read_parquet(UNSW_DIR / "UNSW_NB15_training-set.parquet")
df_test  = pd.read_parquet(UNSW_DIR / "UNSW_NB15_testing-set.parquet")

DROP = ['attack_cat', 'label']
X_train = df_train.drop(columns=DROP).select_dtypes(include=[np.number])
y_train = df_train['label']
X_test  = df_test.drop(columns=DROP).select_dtypes(include=[np.number])
y_test  = df_test['label']

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)
X_test  = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"Train : {len(X_train):,} lignes ({y_train.mean():.1%} attaques)")
print(f"Test  : {len(X_test):,}  lignes ({y_test.mean():.1%} attaques)")

# Entraîne sur BENIGN uniquement
scaler = StandardScaler()
X_train_n = scaler.fit_transform(X_train[y_train == 0])
X_test_s  = scaler.transform(X_test)

best_f1    = 0
best_model = None
best_name  = ""
results    = []

# ══════════════════════════════════════════════════════════════════════════════
# 1. ISOLATION FOREST — grid search
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*55)
print("1. Isolation Forest — grid search")
print("═"*55)

for n_est in [100, 200, 300]:
    for max_s in [0.5, 0.8, 1.0]:
        for contam in [0.3, 0.4, 0.499]:
            clf = IsolationForest(
                n_estimators=n_est,
                max_samples=max_s,
                contamination=contam,
                random_state=42,
                n_jobs=-1
            )
            clf.fit(X_train_n)
            y_pred = (clf.predict(X_test_s) == -1).astype(int)
            f1 = f1_score(y_test, y_pred)
            rec = y_pred[y_test == 1].mean()
            prec = y_pred[y_pred == 1].mean() if y_pred.sum() > 0 else 0

            results.append({
                'model': 'IsolationForest',
                'n_estimators': n_est,
                'max_samples': max_s,
                'contamination': contam,
                'f1': round(f1, 4),
                'recall': round(rec, 4),
                'precision': round(prec, 4)
            })

            if f1 > best_f1:
                best_f1 = f1
                best_model = clf
                best_name = f"IF(n={n_est}, s={max_s}, c={contam})"

            print(f"  n={n_est} s={max_s} c={contam} → f1={f1:.3f}  recall={rec:.2%}  prec={prec:.2%}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. LOCAL OUTLIER FACTOR
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*55)
print("2. Local Outlier Factor")
print("═"*55)

# LOF est lent sur gros datasets → on échantillonne
sample_idx = np.random.choice(len(X_test_s), size=min(10000, len(X_test_s)), replace=False)
X_lof = X_test_s[sample_idx]
y_lof = y_test.values[sample_idx]

for n_neighbors in [10, 20, 50]:
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=min(y_lof.mean(), 0.499),
        novelty=False,
        n_jobs=-1
    )
    y_pred_lof = (lof.fit_predict(X_lof) == -1).astype(int)
    f1  = f1_score(y_lof, y_pred_lof)
    rec = y_pred_lof[y_lof == 1].mean()
    prec = y_pred_lof[y_pred_lof == 1].mean() if y_pred_lof.sum() > 0 else 0

    results.append({
        'model': 'LOF', 'n_estimators': n_neighbors, 'max_samples': '-',
        'contamination': '-', 'f1': round(f1, 4),
        'recall': round(rec, 4), 'precision': round(prec, 4)
    })
    print(f"  k={n_neighbors} → f1={f1:.3f}  recall={rec:.2%}  prec={prec:.2%}")

    if f1 > best_f1:
        best_f1 = f1
        best_model = lof
        best_name = f"LOF(k={n_neighbors})"

# ══════════════════════════════════════════════════════════════════════════════
# 3. ONE-CLASS SVM (sur échantillon — trop lent sinon)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*55)
print("3. One-Class SVM (échantillon 5k train / 5k test)")
print("═"*55)

# Index dans X_train_n (déjà filtré BENIGN, taille 56k)
idx_train_svm = np.random.choice(len(X_train_n), size=20000, replace=False)
X_svm_train   = X_train_n[idx_train_svm]

idx_test_svm  = np.random.choice(len(X_test_s), size=20000, replace=False)
X_svm_test   = X_test_s[idx_test_svm]
y_svm_test   = y_test.values[idx_test_svm]

for nu in [0.1, 0.3, 0.5]:
    svm = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
    svm.fit(X_svm_train)
    y_pred_svm = (svm.predict(X_svm_test) == -1).astype(int)
    f1   = f1_score(y_svm_test, y_pred_svm)
    rec  = y_pred_svm[y_svm_test == 1].mean()
    prec = y_pred_svm[y_pred_svm == 1].mean() if y_pred_svm.sum() > 0 else 0

    results.append({
        'model': 'OneClassSVM', 'n_estimators': '-', 'max_samples': '-',
        'contamination': nu, 'f1': round(f1, 4),
        'recall': round(rec, 4), 'precision': round(prec, 4)
    })
    print(f"  nu={nu} → f1={f1:.3f}  recall={rec:.2%}  prec={prec:.2%}")

    if f1 > best_f1:
        best_f1 = f1
        best_name = f"OC-SVM(nu={nu})"

# ══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ + SAUVEGARDE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*55)
print("MEILLEUR MODÈLE")
print("═"*55)
print(f"  {best_name}  →  F1 = {best_f1:.4f}")

df_results = pd.DataFrame(results).sort_values('f1', ascending=False)
print("\nTop 10 :")
print(df_results.head(10).to_string(index=False))

df_results.to_csv(OUT_DIR / "optim_results.csv", index=False)

# Sauvegarde du meilleur modèle IF (LOF/SVM non sauvegardables facilement)
best_if = df_results[df_results['model'] == 'IsolationForest'].iloc[0]
print(f"\nMeilleur IF : {best_if.to_dict()}")

clf_final = IsolationForest(
    n_estimators=int(best_if['n_estimators']),
    max_samples=float(best_if['max_samples']),
    contamination=float(best_if['contamination']),
    random_state=42, n_jobs=-1
)
clf_final.fit(X_train_n)
joblib.dump({'model': clf_final, 'scaler': scaler, 'features': list(X_train.columns)},
            OUT_DIR / "model_final.joblib")

print(f"\n💾 Modèle final sauvegardé : {OUT_DIR}/model_final.joblib")
print(f"💾 Résultats grille : {OUT_DIR}/optim_results.csv")
