import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Password Spray Detector", layout="wide")

st.title("Password Spray Detector (features + anomalies)")

scored_path = st.text_input("Chemin scored.csv", "data/processed/scored.csv")
alerts_path = st.text_input("Chemin alerts.csv", "data/processed/alerts.csv")

try:
    scored = pd.read_csv(scored_path)
    alerts = pd.read_csv(alerts_path)
except Exception as e:
    st.error(f"Impossible de lire les CSV: {e}")
    st.stop()

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("# fenêtres", len(scored))
with c2:
    st.metric("# alertes", int((scored.get('is_anomaly', pd.Series([0]*len(scored)))==1).sum()))
with c3:
    st.metric("Score max", float(scored['anomaly_score'].max()))

st.subheader("Top alertes")
st.dataframe(alerts.head(50), use_container_width=True)

st.subheader("Scores")
fig = px.histogram(scored, x="anomaly_score", nbins=60)
st.plotly_chart(fig, use_container_width=True)

if "ts" in scored.columns:
    st.subheader("Anomalies dans le temps")
    tmp = scored.copy()
    tmp["ts"] = pd.to_datetime(tmp["ts"], utc=True)
    by_t = tmp.groupby(pd.Grouper(key="ts", freq="30min"))["is_anomaly"].sum().reset_index()
    fig2 = px.line(by_t, x="ts", y="is_anomaly")
    st.plotly_chart(fig2, use_container_width=True)
