# Password Spray Detector (IA + SOC)

Projet: détection d'anomalies d'authentification (password spraying / brute force / outage) à partir de logs, avec extraction de features, modèle IsolationForest, génération d'alertes, et une app de visualisation.

## Démarrage rapide

```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

# 1) Générer un dataset synthétique avancé
python -m src.cli synth --out data/raw/auth_logs.csv --minutes 720

# 2) Extraire features + entraîner + scorer + exporter
python -m src.cli run --in data/raw/auth_logs.csv --outdir data/processed --window 10min

# 3) Voir les top alertes
python -m src.cli top --alerts data/processed/alerts.csv --n 20

# (Optionnel) dashboard
pip install -r requirements-app.txt
streamlit run src/dashboard_streamlit.py
```

## Format attendu des logs (CSV)
Colonnes minimales:
- ts (timestamp ISO)
- user
- src_ip
- app
- result  (success/fail)

Colonnes optionnelles:
- reason, user_agent, country

## Datasets publics
Le code inclut des "adapters" (stubs) pour mapper des datasets publics vers ce schéma (à compléter quand tu auras téléchargé les données).
