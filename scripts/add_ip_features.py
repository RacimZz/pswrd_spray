import pandas as pd
import numpy as np

scored = pd.read_csv("data/processed/rba_1h/scored.csv")
scored["ts"] = pd.to_datetime(scored["ts"], utc=True)
scored = scored.sort_values(["src_ip", "ts"]).reset_index(drop=True)

# 1. Nombre de fenêtres actives par IP
scored["n_windows_active"] = scored.groupby("src_ip")["ts"].transform("count")

# 2. Nombre total d'utilisateurs distincts ciblés par IP (sur tout l'historique)
scored["n_distinct_users_total"] = scored.groupby("src_ip")["n_users"].transform("sum")

# 3. Taux d'échec global de l'IP (moyenne sur toutes ses fenêtres)
scored["fail_rate_global"] = scored.groupby("src_ip")["fail_rate"].transform("mean")

# 4. Age de l'IP : heures depuis la première apparition
first_seen = scored.groupby("src_ip")["ts"].transform("min")
scored["ip_age_hours"] = (scored["ts"] - first_seen).dt.total_seconds() / 3600

# 5. Variance du nombre de tentatives (IP régulière vs burst)
scored["n_attempts_std"] = scored.groupby("src_ip")["n_attempts"].transform("std").fillna(0)

scored.to_csv("data/processed/rba_1h/scored_enriched.csv", index=False)
print(f"Features ajoutées : {len(scored)} lignes")
print(scored[["n_windows_active","n_distinct_users_total","fail_rate_global","ip_age_hours"]].describe().round(2))
