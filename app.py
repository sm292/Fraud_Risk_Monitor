"""Streamlit dashboard for retrospective benchmark monitoring."""
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
SCORES, METRICS = ROOT / "data/processed/scored_transactions.csv", ROOT / "artifacts/metrics.json"
st.set_page_config(page_title="Fraud Risk Monitor", page_icon="🛡️", layout="wide")
st.title("Transaction Risk Monitor")
st.caption("Retrospective benchmark demonstration. Scores are not production decisions.")
if not SCORES.exists() or not METRICS.exists():
    st.warning("No scored benchmark yet. Follow the setup and training commands in README.md.")
    st.stop()
df, metrics = pd.read_csv(SCORES), json.loads(METRICS.read_text(encoding="utf-8"))
st.sidebar.header("Investigation assumptions")
recovery = st.sidebar.slider("Recoverable share of flagged fraud amount", 0.0, 1.0, 0.50, 0.05)
cost = st.sidebar.number_input("Cost per investigation ($)", min_value=0.0, value=25.0, step=5.0)
floor = st.sidebar.slider("Minimum risk score", 0.0, 1.0, float(metrics["threshold"]), 0.001)
queue = df[df.flagged & (df.risk_score >= floor)].copy()
gross = float(queue.loc[queue.Class == 1, "Amount"].sum()) * recovery
expense = len(queue) * cost
c1,c2,c3,c4 = st.columns(4)
c1.metric("Test-period transactions", f"{len(df):,}")
c2.metric("Investigation queue", f"{len(queue):,}")
c3.metric("Test average precision", f"{metrics['test_average_precision']:.3f}")
c4.metric("Illustrative net loss avoided", f"${gross-expense:,.2f}")
st.caption(f"Scenario: gross recoveries ${gross:,.2f} less investigation cost ${expense:,.2f}. Assumes Amount is a monetary unit, recoveries apply uniformly only to flagged fraud, and every queue item is investigated. This is not measured or causal savings.")
left,right=st.columns(2)
with left:
    st.subheader("Risk score distribution")
    st.plotly_chart(px.histogram(df, x="risk_score", color="Class", nbins=40, barmode="overlay", labels={"Class":"Actual label"}), use_container_width=True)
with right:
    st.subheader("Validation model comparison")
    st.plotly_chart(px.bar(pd.DataFrame(metrics["model_comparison"]), x="model", y="validation_average_precision"), use_container_width=True)
st.subheader("Investigation queue")
st.caption("Actual labels are visible only because this is an offline test benchmark; live labels would arrive after investigation.")
queue = queue.sort_values("risk_score", ascending=False)
queue["actual_outcome"] = queue.Class.map({0:"Legitimate", 1:"Fraud"})
st.dataframe(queue[["transaction_id","Time","Amount","risk_score","anomaly_score","review_status","actual_outcome"]], use_container_width=True, hide_index=True)
st.download_button("Download queue CSV", queue.to_csv(index=False), "investigation_queue.csv", "text/csv")
with st.expander("Evaluation details"):
    st.json(metrics)
