import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Charge les données enrichies
rba = pd.read_csv("data/public/rba/rba_1m.csv")
ip_labels = rba.groupby("IP Address")["Is Attack IP"].max().reset_index()
ip_labels.columns = ["src_ip", "is_attack_ip"]

scored = pd.read_csv("data/processed/rba_1h/scored_enriched.csv")
for col in ["is_attack_ip"]:
    if col in scored.columns:
        scored = scored.drop(columns=[col])
scored = scored.merge(ip_labels, on="src_ip", how="left")
scored["is_attack_ip"] = scored["is_attack_ip"].fillna(0).astype(int)

# Features étendues
FEATURE_COLS = [
    "n_attempts", "n_fail", "n_success", "n_users", "n_apps",
    "n_user_agents", "n_countries", "fail_rate", "attempts_per_min",
    "success_after_fail", "user_entropy",
    "n_windows_active", "n_distinct_users_total",
    "fail_rate_global", "ip_age_hours", "n_attempts_std",
]

X = scored[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
y_true = scored["is_attack_ip"]

# Réentraîne IsolationForest avec les nouvelles features
scaler = StandardScaler()
Xn = scaler.fit_transform(X)

model = IsolationForest(n_estimators=300, contamination=0.01, random_state=42, n_jobs=-1)
model.fit(Xn)
y_pred = (model.predict(Xn) == -1).astype(int)

print("=== Matrice de confusion ===")
cm = confusion_matrix(y_true, y_pred)
print(f"                 Prédit Normal  Prédit Attaque")
print(f"Vrai Normal      {cm[0][0]:>13}  {cm[0][1]:>13}")
print(f"Vrai Attaque     {cm[1][0]:>13}  {cm[1][1]:>13}")

print("\n=== Métriques ===")
print(classification_report(y_true, y_pred, target_names=["Normal", "Attaque"], digits=3))

total = len(scored)
fp, fn = cm[0][1], cm[1][0]
print(f"Taux d'erreur global : {(fp + fn) / total:.2%}")
print(f"Faux positifs (FP)   : {fp} ({fp/total:.2%})")
print(f"Faux négatifs (FN)   : {fn} ({fn/total:.2%})")
