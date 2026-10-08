import io
import json
import os

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, average_precision_score,
)

from core.config import settings
from core.model_loader import load_pipeline, load_sample_data, get_feature_schema
from core.feature_engineering import engineer_features
from core.inputs import render_input_form
from core.shap_utils import explain_instance, explain_batch, top_contributing_features
from core.threshold_optimizer import compute_cost_curve, find_optimal_threshold
from core.retention_alerts import build_retention_message, build_alerts_for_batch
from core.drift_detection import compute_drift_report, PSI_SIGNIFICANT_THRESHOLD
from core.monitoring_log import log_batch, load_log

st.set_page_config(page_title="Churn Prediction Dashboard", page_icon="📉", layout="wide")

pipeline = load_pipeline()
sample_df = load_sample_data()
schema = get_feature_schema(sample_df)

if "churn_threshold" not in st.session_state:
    st.session_state.churn_threshold = settings.CHURN_THRESHOLD

st.title("📉 Customer Churn Prediction Dashboard")
st.caption(f"Current decision threshold: **{st.session_state.churn_threshold:.2f}** "
           f"(set in the 💰 Cost-Sensitive Threshold tab)")

tab_whatif, tab_batch, tab_threshold, tab_monitoring, tab_performance = st.tabs(
    ["🔮 What-If Analysis", "📂 Batch Prediction", "💰 Cost-Sensitive Threshold",
     "📡 Model Monitoring", "📊 Model Performance"]
)

# ---------------------------------------------------------------------------
# Tab 1: What-If Analysis
# ---------------------------------------------------------------------------
with tab_whatif:
    st.subheader("Predict churn risk for a single customer")
    st.caption("Adjust the inputs below and see how the predicted risk changes in real time.")

    raw_input_df = render_input_form(schema)

    if st.button("Predict Churn Risk", type="primary"):
        engineered_df = engineer_features(raw_input_df)

        proba = pipeline.predict_proba(engineered_df)[0, 1]
        is_high_risk = proba >= st.session_state.churn_threshold

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Churn Probability", f"{proba:.1%}")
            if proba >= 0.6:
                st.error("🔴 High risk")
            elif proba >= 0.35:
                st.warning("🟡 Medium risk")
            else:
                st.success("🟢 Low risk")

        with col2:
            st.markdown("**Top factors driving this prediction:**")
            background_sample = sample_df.drop(columns=["Churn"]).sample(
                min(100, len(sample_df)), random_state=42
            )
            shap_row, _ = explain_instance(pipeline, engineered_df, background_sample)
            top_features = top_contributing_features(shap_row, top_n=5)
            st.dataframe(top_features, use_container_width=True, hide_index=True)

        with st.expander("📈 SHAP waterfall plot (detailed)"):
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_row, show=False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        if proba >= 0.35:
            st.markdown("**📧 Suggested retention outreach**")
            risk_level = "High" if proba >= 0.6 else "Medium"
            message = build_retention_message(
                raw_input_df.iloc[0].to_dict(), proba, risk_level
            )
            st.text_input("Subject", value=message["subject"], disabled=True)
            st.text_area("Body", value=message["body"], height=220)


# ---------------------------------------------------------------------------
# Tab 2: Batch Prediction
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Predict churn risk for many customers at once")
    st.caption(
        "Upload a CSV with the same raw columns as the training data "
        "(no need to include the derived tenure_group/num_addon_services/"
        "avg_monthly_spend columns — those are computed automatically)."
    )

    uploaded_csv = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded_csv is not None:
        batch_df = pd.read_csv(uploaded_csv)
        st.write(f"Loaded {len(batch_df)} rows.")
        st.dataframe(batch_df.head(), use_container_width=True)

        if st.button("Run Batch Prediction", type="primary"):
            with st.spinner("Scoring customers..."):
                engineered_batch = engineer_features(batch_df)
                probas = pipeline.predict_proba(engineered_batch)[:, 1]

                result_df = batch_df.copy()
                result_df["churn_probability"] = probas
                result_df["predicted_churn"] = pd.Series(
                    probas >= st.session_state.churn_threshold, index=result_df.index
                ).map({True: "Yes", False: "No"})

            st.success(f"Scored {len(result_df)} customers. "
                       f"{(result_df['predicted_churn'] == 'Yes').sum()} flagged as likely to churn.")
            st.dataframe(
                result_df.sort_values("churn_probability", ascending=False),
                use_container_width=True,
            )

            csv_buffer = io.StringIO()
            result_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "⬇️ Download predictions as CSV",
                data=csv_buffer.getvalue(),
                file_name="churn_predictions.csv",
                mime="text/csv",
            )

            st.session_state.last_batch_result = result_df
            st.session_state.last_batch_raw = batch_df  # for drift comparison

            drift_report = compute_drift_report(sample_df, batch_df)
            max_psi = drift_report["psi"].max() if not drift_report.empty else 0.0
            drifted_features = int((drift_report["psi"] >= PSI_SIGNIFICANT_THRESHOLD).sum())

            log_batch(
                n_rows=len(result_df),
                mean_probability=result_df["churn_probability"].mean(),
                n_flagged_high_risk=int((result_df["predicted_churn"] == "Yes").sum()),
                max_psi=max_psi,
                drifted_features=drifted_features,
            )
            st.session_state.last_drift_report = drift_report

            if drifted_features > 0:
                st.warning(
                    f"⚠️ {drifted_features} feature(s) show significant drift vs. the "
                    f"training data — see the 📡 Model Monitoring tab for details."
                )

    if "last_batch_result" in st.session_state:
        st.markdown("---")
        st.subheader("📧 Automated Retention Alerts")
        st.caption(
            "Instead of just reviewing a table, generate a ready-to-send "
            "retention message for every customer above a risk cutoff."
        )

        alert_threshold = st.slider(
            "Generate alerts for customers at or above this churn probability",
            0.0, 1.0, 0.6, 0.05,
        )

        if st.button("Generate Retention Alerts"):
            alerts_df = build_alerts_for_batch(
                st.session_state.last_batch_result, threshold=alert_threshold
            )
            st.session_state.last_alerts = alerts_df

        if "last_alerts" in st.session_state:
            alerts_df = st.session_state.last_alerts
            st.write(f"{len(alerts_df)} customers flagged for outreach.")

            for _, row in alerts_df.head(20).iterrows():
                label = f"{row.get('customerID', 'Customer')} — " \
                        f"{row['churn_probability']:.1%} risk — {row['offer_type']}"
                with st.expander(label):
                    st.text(f"Subject: {row['message_subject']}")
                    st.text_area(
                        "Body", value=row["message_body"], height=180,
                        key=f"body_{row.name}",
                    )

            if len(alerts_df) > 20:
                st.caption(f"Showing the first 20 of {len(alerts_df)} — download all below.")

            alerts_csv = io.StringIO()
            alerts_df.to_csv(alerts_csv, index=False)
            st.download_button(
                "⬇️ Download all retention alerts as CSV",
                data=alerts_csv.getvalue(),
                file_name="retention_alerts.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# Tab 3: Cost-Sensitive Threshold
# ---------------------------------------------------------------------------
with tab_threshold:
    st.subheader("Find the threshold that minimizes real business cost")
    st.caption(
        "A fixed 0.5 cutoff treats a missed churner and a wasted retention "
        "offer as equally costly — they usually aren't. Enter your real "
        "costs below to find the threshold that minimizes total cost on "
        "the test set."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        cost_fn = st.number_input(
            "Cost of missing a churner (false negative) — e.g. lost revenue $",
            min_value=0.0, value=500.0, step=50.0,
        )
    with col_b:
        cost_fp = st.number_input(
            "Cost of a wasted retention offer (false positive) — $",
            min_value=0.0, value=50.0, step=10.0,
        )

    if st.button("Optimize Threshold", type="primary"):
        X_test = sample_df.drop(columns=["Churn"])
        y_test = sample_df["Churn"]
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        cost_curve = compute_cost_curve(y_test, y_proba, cost_fn, cost_fp)
        best = find_optimal_threshold(cost_curve)
        default_row = cost_curve.iloc[(cost_curve["threshold"] - 0.5).abs().idxmin()]

        st.session_state.last_cost_curve = cost_curve
        st.session_state.last_best_threshold = best
        st.session_state.last_default_cost = default_row["total_cost"]

    # Rendered from session_state (not gated behind the button click above) so
    # results stay visible even after the "Apply" button below triggers a rerun.
    if "last_best_threshold" in st.session_state:
        cost_curve = st.session_state.last_cost_curve
        best = st.session_state.last_best_threshold
        default_cost = st.session_state.last_default_cost

        col1, col2, col3 = st.columns(3)
        col1.metric("Optimal threshold", f"{best['threshold']:.2f}")
        col2.metric("Total cost at optimal", f"${best['total_cost']:,.0f}")
        col3.metric(
            "Total cost at 0.5 (default)",
            f"${default_cost:,.0f}",
            delta=f"${best['total_cost'] - default_cost:,.0f}",
            delta_color="inverse",
        )

        st.markdown("**Total cost across thresholds**")
        chart_df = cost_curve.set_index("threshold")[["total_cost"]]
        st.line_chart(chart_df)

        st.markdown("**Breakdown at the optimal threshold**")
        st.dataframe(
            pd.DataFrame([{
                "False Negatives (missed churners)": int(best["false_negatives"]),
                "False Positives (wasted offers)": int(best["false_positives"]),
                "True Positives (correctly caught)": int(best["true_positives"]),
                "True Negatives (correctly ignored)": int(best["true_negatives"]),
            }]),
            use_container_width=True, hide_index=True,
        )

        if st.button(f"✅ Apply threshold {best['threshold']:.2f} to the dashboard"):
            st.session_state.churn_threshold = float(best["threshold"])
            st.success(
                f"Threshold set to {best['threshold']:.2f} — this now applies to "
                f"the What-If and Batch Prediction tabs."
            )
            st.rerun()


# ---------------------------------------------------------------------------
# Tab 4: Model Monitoring
# ---------------------------------------------------------------------------
with tab_monitoring:
    st.subheader("Is incoming data starting to look different from training data?")
    st.caption(
        "A model's accuracy silently degrades when the customers it sees "
        "in production start to differ from what it was trained on — this "
        "is called data drift. Every batch prediction you run is compared "
        "against the test set automatically."
    )

    if "last_drift_report" not in st.session_state:
        st.info(
            "👈 Run a batch prediction in the 📂 Batch Prediction tab first — "
            "drift is measured against whatever CSV you upload there."
        )
    else:
        drift_report = st.session_state.last_drift_report
        n_drifted = int((drift_report["psi"] >= PSI_SIGNIFICANT_THRESHOLD).sum())

        if n_drifted > 0:
            st.error(f"🔴 {n_drifted} feature(s) show significant drift — "
                      f"consider retraining on recent data (see the API's /retrain endpoint).")
        else:
            st.success("🟢 No significant drift detected in the last uploaded batch.")

        st.markdown("**Drift by feature (Population Stability Index)**")
        st.dataframe(drift_report, use_container_width=True, hide_index=True)
        st.caption(
            "PSI < 0.10 = stable · 0.10–0.25 = moderate drift, worth watching · "
            "≥ 0.25 = significant drift"
        )

        st.markdown("**PSI by feature**")
        chart_df = drift_report.set_index("feature")[["psi"]]
        st.bar_chart(chart_df)

    st.markdown("---")
    st.subheader("Prediction trend over time")
    st.caption(
        "Every batch prediction is logged locally, so you can see whether "
        "average predicted risk is drifting up or down across sessions."
    )

    history = load_log()
    if history.empty:
        st.info("No batch predictions logged yet.")
    else:
        st.line_chart(history.set_index("timestamp")[["mean_churn_probability"]])
        st.dataframe(
            history.sort_values("timestamp", ascending=False),
            use_container_width=True, hide_index=True,
        )


# ---------------------------------------------------------------------------
# Tab 5: Model Performance
# ---------------------------------------------------------------------------
with tab_performance:
    st.subheader("Model performance on the held-out test set")
    st.caption(f"Using decision threshold: {st.session_state.churn_threshold:.2f}")

    test_df = sample_df  # loaded from SAMPLE_DATA_PATH, expected to be test.csv
    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= st.session_state.churn_threshold).astype(int)

    col1, col2 = st.columns(2)
    col1.metric("F1-score (Churn class)", f"{f1_score(y_test, y_pred):.3f}")
    col2.metric("AUC-PR", f"{average_precision_score(y_test, y_proba):.3f}")

    st.markdown("**Classification report**")
    report = classification_report(
        y_test, y_pred, target_names=["No Churn", "Churn"], output_dict=True
    )
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: No Churn", "Actual: Churn"],
        columns=["Predicted: No Churn", "Predicted: Churn"],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("**Global feature importance (SHAP)**")
    with st.spinner("Computing SHAP values for the test set..."):
        # Sample up to 200 rows for speed on larger test sets
        sample_for_shap = X_test.sample(min(200, len(X_test)), random_state=42)
        background_sample = X_test.sample(min(100, len(X_test)), random_state=1)
        shap_values, _ = explain_batch(pipeline, sample_for_shap, background_sample)

    fig, ax = plt.subplots()
    shap.plots.beeswarm(shap_values, show=False, max_display=12)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Show the tuning comparison table too, if it was saved by tune_models.py
    tuning_report_path = os.path.join(
        os.path.dirname(settings.MODEL_PATH), "tuning_report.json"
    )
    if os.path.exists(tuning_report_path):
        with open(tuning_report_path) as f:
            tuning_report = json.load(f)
        st.markdown("**Model comparison (from tuning step)**")
        st.json(tuning_report["all_results"])
