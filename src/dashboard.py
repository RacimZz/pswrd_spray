from __future__ import annotations

"""Dashboard Streamlit - Password Spray Detector.

Lancement:
    streamlit run src/dashboard.py
"""

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Config page
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Password Spray Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("Password Spray Detector")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Pipeline", "Resultats", "Detail alerte", "Comparaison features", "SHAP global"],
)

st.sidebar.markdown("---")

dataset_label = st.sidebar.selectbox(
    "Dataset",
    ["Synthetique", "RBA (public)", "LANL (public)"],
)

DATASET_PATHS = {
    "Synthetique":   "data/processed",
    "RBA (public)":  "data/processed/rba",
    "LANL (public)": "data/processed/lanl",
}
base_dir = Path(DATASET_PATHS[dataset_label])

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def safe_load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return load_csv(str(path))


def parse_ts(df: pd.DataFrame) -> pd.DataFrame:
    if df is not None and "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df


def run_cmd(cmd: list[str], label: str, progress_bar, step: float) -> bool:
    """Execute une commande subprocess et affiche le resultat dans Streamlit."""
    with st.expander(f"[{label}]", expanded=True):
        placeholder = st.empty()
        placeholder.info(f"En cours : {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            placeholder.success(f"Termine : {label}")
            if result.stdout.strip():
                st.code(result.stdout[-2000:], language="bash")
        else:
            placeholder.error(f"Erreur : {label}")
            st.code(result.stderr[-2000:], language="bash")
            return False
    progress_bar.progress(step)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# PAGE : Pipeline
# ─────────────────────────────────────────────────────────────────────────────

if page == "Pipeline":
    st.title("Lancer le pipeline")
    st.markdown(
        "Configure les parametres et lance toutes les etapes en un clic. "
        "Les resultats seront disponibles dans l'onglet **Resultats**."
    )

    st.markdown("---")
    st.subheader("1. Source des logs")

    source = st.radio("Type de source", ["Generer des logs synthetiques", "Charger un CSV existant"])

    if source == "Generer des logs synthetiques":
        col1, col2, col3 = st.columns(3)
        minutes   = col1.slider("Duree (minutes)", 60, 1440, 720, step=60)
        n_users   = col2.slider("Nombre d'utilisateurs", 50, 1000, 400, step=50)
        n_ips     = col3.slider("Nombre d'IPs normales", 10, 200, 80, step=10)
        raw_path  = "data/raw/auth_logs.csv"

    else:
        uploaded = st.file_uploader("Charger un fichier CSV de logs", type=["csv"])
        raw_path = "data/raw/uploaded_logs.csv"
        if uploaded is not None:
            Path("data/raw").mkdir(parents=True, exist_ok=True)
            Path(raw_path).write_bytes(uploaded.read())
            st.success(f"Fichier charge : {raw_path}")

    st.markdown("---")
    st.subheader("2. Parametres du pipeline")

    col1, col2, col3 = st.columns(3)
    window        = col1.selectbox("Fenetre d'agregation", ["5min", "10min", "15min", "30min"], index=1)
    contamination = col2.slider("Contamination (taux d'anomalies attendu)", 0.001, 0.05, 0.01, step=0.001)
    n_estimators  = col3.slider("Nombre d'arbres IsolationForest", 100, 500, 300, step=50)

    st.markdown("---")
    st.subheader("3. Post-filtrage")

    col1, col2, col3 = st.columns(3)
    min_attempts  = col1.number_input("Min tentatives", 1, 200, 20)
    min_fail      = col2.number_input("Min echecs", 1, 100, 10)
    min_fail_rate = col3.slider("Min taux d'echec", 0.0, 1.0, 0.30, step=0.05)

    st.markdown("---")
    st.subheader("4. SHAP")
    run_shap = st.checkbox("Calculer les explications SHAP (plus lent)", value=True)

    st.markdown("---")

    if st.button("Lancer le pipeline complet", type="primary"):

        outdir = str(base_dir)
        progress = st.progress(0.0)
        st.markdown("**Progression**")
        success = True

        # Etape 1 : generation synthetique si besoin
        if source == "Generer des logs synthetiques":
            success = run_cmd(
                [sys.executable, "-m", "src.cli", "synth",
                 "--out", raw_path,
                 "--minutes", str(minutes),
                 "--n-users", str(n_users),
                 "--n-normal-ips", str(n_ips)],
                "Generation des logs synthetiques",
                progress, 0.15,
            )

        # Etape 2 : pipeline features + modele + scoring
        if success:
            success = run_cmd(
                [sys.executable, "-m", "src.cli", "run",
                 "--in", raw_path,
                 "--outdir", outdir,
                 "--window", window,
                 "--contamination", str(contamination),
                 "--n-estimators", str(n_estimators)],
                "Features + IsolationForest + Scoring",
                progress, 0.50,
            )

        # Etape 3 : post-filtrage
        if success:
            success = run_cmd(
                [sys.executable, "-m", "src.cli_postfilter",
                 "--alerts", str(base_dir / "alerts.csv"),
                 "--outdir", outdir,
                 "--min-attempts", str(int(min_attempts)),
                 "--min-fail", str(int(min_fail)),
                 "--min-fail-rate", str(min_fail_rate)],
                "Post-filtrage des alertes",
                progress, 0.75,
            )

        # Etape 4 : SHAP (optionnel)
        if success and run_shap:
            shap_outdir = str(base_dir / "shap")
            model_path  = str(base_dir / "model_iforest.joblib")
            scored_path = str(base_dir / "scored.csv")
            deduped_path= str(base_dir / "alerts_deduped.csv")

            success = run_cmd(
                [sys.executable, "-m", "src.cli_shap",
                 "--model",  model_path,
                 "--scored", scored_path,
                 "--alerts", deduped_path,
                 "--outdir", shap_outdir],
                "Explications SHAP",
                progress, 0.95,
            )

        progress.progress(1.0)

        if success:
            st.success("Pipeline termine. Va dans l'onglet Resultats pour voir les alertes.")
            st.cache_data.clear()
        else:
            st.error("Le pipeline s'est arrete a cause d'une erreur (voir details ci-dessus).")


# ─────────────────────────────────────────────────────────────────────────────
# Chargement commun pour les pages de resultats
# ─────────────────────────────────────────────────────────────────────────────

else:
    scored   = parse_ts(safe_load(base_dir / "scored.csv"))
    alerts   = parse_ts(safe_load(base_dir / "alerts_deduped.csv"))
    features = parse_ts(safe_load(base_dir / "features.csv"))
    shap_dir = base_dir / "shap"

    if scored is None or alerts is None:
        st.warning(
            f"Aucun resultat trouve dans `{base_dir}`. "
            "Lance d'abord le pipeline depuis l'onglet **Pipeline**."
        )
        st.stop()

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE : Resultats
    # ─────────────────────────────────────────────────────────────────────────

    if page == "Resultats":
        st.title("Resultats")

        n_anomalies   = int(scored["is_anomaly"].sum()) if "is_anomaly" in scored.columns else 0
        n_alerts      = len(alerts)
        n_critical    = int((alerts["priority"] == "CRITICAL").sum()) if "priority" in alerts.columns else 0
        n_high        = int((alerts["priority"] == "HIGH").sum()) if "priority" in alerts.columns else 0
        n_compromised = int((alerts["success_after_fail"] == 1).sum()) if "success_after_fail" in alerts.columns else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Fenetres analysees", f"{len(scored):,}")
        c2.metric("Anomalies brutes",   f"{n_anomalies:,}")
        c3.metric("Alertes finales",    f"{n_alerts}")
        c4.metric("CRITICAL",           f"{n_critical}")
        c5.metric("Comptes compromis?", f"{n_compromised}")

        st.markdown("---")

        col_tl, col_hist = st.columns([2, 1])

        with col_tl:
            st.markdown("**Timeline des anomalies (par 30 min)**")
            by_time = (
                scored.dropna(subset=["ts"])
                      .set_index("ts")
                      .resample("30min")["is_anomaly"]
                      .sum()
                      .reset_index()
            )
            fig_tl = px.area(by_time, x="ts", y="is_anomaly",
                             labels={"ts": "Heure", "is_anomaly": "Nombre d'anomalies"},
                             color_discrete_sequence=["#e74c3c"])
            fig_tl.update_layout(margin=dict(t=10, b=10), height=260)
            st.plotly_chart(fig_tl, use_container_width=True)

        with col_hist:
            st.markdown("**Distribution des scores**")
            fig_h = px.histogram(scored, x="anomaly_score", nbins=60,
                                 color_discrete_sequence=["#3498db"])
            fig_h.update_layout(margin=dict(t=10, b=10), height=260, showlegend=False)
            st.plotly_chart(fig_h, use_container_width=True)

        st.markdown("---")
        st.markdown("**Table des alertes**")

        if "priority" in alerts.columns:
            prio_filter = st.multiselect(
                "Filtrer par priorite",
                ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                default=["CRITICAL", "HIGH"],
            )
            disp = alerts[alerts["priority"].isin(prio_filter)]
        else:
            disp = alerts

        cols = [c for c in ["priority", "src_ip", "ts", "anomaly_score",
                             "n_attempts", "n_fail", "n_users", "fail_rate",
                             "attempts_per_min", "success_after_fail"] if c in disp.columns]
        st.dataframe(disp[cols].reset_index(drop=True), use_container_width=True, height=300)

        st.markdown("---")
        st.download_button(
            "Telecharger les alertes (CSV)",
            data=alerts.to_csv(index=False),
            file_name="alerts_final.csv",
            mime="text/csv",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE : Detail alerte
    # ─────────────────────────────────────────────────────────────────────────

    elif page == "Detail alerte":
        st.title("Detail d'une alerte")

        if alerts.empty:
            st.info("Aucune alerte disponible.")
            st.stop()

        options = [
            f"{row.get('priority','?')} | {row['src_ip']} | {str(row['ts'])[:16]}"
            for _, row in alerts.iterrows()
        ]
        selected = st.selectbox("Selectionne une alerte", options)
        sel_idx  = options.index(selected)
        sel_row  = alerts.iloc[sel_idx]

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Metriques de l'alerte**")
            m = {
                "IP source":          sel_row.get("src_ip", "?"),
                "Fenetre":            str(sel_row.get("ts", "?"))[:19],
                "Priorite":           sel_row.get("priority", "?"),
                "Score d'anomalie":   f"{float(sel_row.get('anomaly_score', 0)):.4f}",
                "Tentatives":         int(sel_row.get("n_attempts", 0)),
                "Echecs":             int(sel_row.get("n_fail", 0)),
                "Taux d'echec":       f"{float(sel_row.get('fail_rate', 0)):.1%}",
                "Comptes cibles":     int(sel_row.get("n_users", 0)),
                "Tentatives/min":     f"{float(sel_row.get('attempts_per_min', 0)):.1f}",
                "Succes apres echec": "OUI" if int(sel_row.get("success_after_fail", 0)) == 1 else "Non",
                "Entropie users":     f"{float(sel_row.get('user_entropy', 0)):.2f}",
            }
            for k, v in m.items():
                st.markdown(f"- **{k}** : `{v}`")

        with c2:
            ip_safe  = str(sel_row.get("src_ip", "")).replace(".", "_")
            ts_safe  = str(sel_row.get("ts", ""))[:16].replace(":", "").replace(" ", "_").replace("+", "")
            fig_path = shap_dir / "shap_waterfalls" / f"alert_{ip_safe}_{ts_safe}.png"

            if fig_path.exists():
                st.markdown("**Explication SHAP**")
                st.image(str(fig_path), use_container_width=True)
            else:
                st.info(
                    "Figure SHAP non disponible.\n\n"
                    "Coche 'Calculer les explications SHAP' dans l'onglet Pipeline et relance."
                )

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE : Comparaison features
    # ─────────────────────────────────────────────────────────────────────────

    elif page == "Comparaison features":
        st.title("Comparaison : trafic normal vs alertes")

        if features is None:
            st.warning("features.csv non trouve.")
            st.stop()

        alert_ips    = set(alerts["src_ip"].tolist()) if "src_ip" in alerts.columns else set()
        feat_normal  = features[~features["src_ip"].isin(alert_ips)]
        feat_attack  = features[features["src_ip"].isin(alert_ips)]

        feat_col = st.selectbox(
            "Feature a comparer",
            [c for c in ["n_attempts", "n_fail", "n_users", "fail_rate",
                         "attempts_per_min", "user_entropy"] if c in features.columns],
        )

        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=feat_normal[feat_col].dropna(), name="Trafic normal",
            marker_color="#3498db", boxpoints="outliers",
        ))
        fig_box.add_trace(go.Box(
            y=feat_attack[feat_col].dropna(), name="IPs suspectes",
            marker_color="#e74c3c", boxpoints="all",
        ))
        fig_box.update_layout(height=400, yaxis_title=feat_col, margin=dict(t=20, b=20))
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("---")
        st.markdown("**Statistiques descriptives**")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("Trafic normal")
            st.dataframe(feat_normal[[feat_col]].describe().round(3), use_container_width=True)
        with c2:
            st.markdown("IPs suspectes")
            st.dataframe(feat_attack[[feat_col]].describe().round(3), use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE : SHAP global
    # ─────────────────────────────────────────────────────────────────────────

    elif page == "SHAP global":
        st.title("Importance globale des features (SHAP)")

        summary_path = shap_dir / "shap_summary.png"
        if summary_path.exists():
            st.image(str(summary_path), width=750)
        else:
            st.info(
                "Lance le pipeline avec SHAP active pour generer ce graphique."
            )

        reports_path = shap_dir / "shap_text_reports.txt"
        if reports_path.exists():
            st.markdown("---")
            st.markdown("**Rapports texte SHAP par alerte**")
            content = reports_path.read_text(encoding="utf-8")
            st.code(content, language="text")

st.sidebar.markdown("---")
st.sidebar.caption("ENSIMAG 1A — Projet perso")
