import pandas as pd

rba = pd.read_csv("data/public/rba/rba_1m.csv")
ip_labels = rba.groupby("IP Address")["Is Attack IP"].max().reset_index()
ip_labels.columns = ["src_ip", "is_attack_ip"]

scored = pd.read_csv("data/processed/rba/scored.csv")

# Supprime les colonnes label si elles existent déjà
for col in ["is_attack_ip", "is_attack_ip_x", "is_attack_ip_y", "is_attack_window"]:
    if col in scored.columns:
        scored = scored.drop(columns=[col])

scored = scored.merge(ip_labels, on="src_ip", how="left")
scored["is_attack_ip"] = scored["is_attack_ip"].fillna(0).astype(int)

scored["is_attack_window"] = (
    (scored["is_attack_ip"] == 1) &
    (scored["n_attempts"] >= 5) &
    (scored["fail_rate"] >= 0.5)
).astype(int)

print(f"Fenetres attaque : {scored['is_attack_window'].sum()} / {len(scored)}")
scored.to_csv("data/processed/rba/scored.csv", index=False)
print("Sauvegarde OK")
