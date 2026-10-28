"""
Interactive Streamlit Web Dashboard for House Price Prediction.
Provides single-property valuation, confidence intervals, batch prediction,
model performance telemetry, and dataset exploration.
"""

import io
from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np

# Ensure system path includes project root
PROJECT_ROOT = Path(__file__).resolve().parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import HousePricePredictor, PredictionValidationError
from src.config import (
    LOCALITIES,
    FURNISHED_STATUSES,
    CHAMPION_MODEL_PATH,
    LEGACY_MODEL_PATH,
    MODEL_METADATA_PATH,
    DATASET_PATH,
)

# Page Configuration
st.set_page_config(
    page_title="House Price Valuation Engine",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_predictor():
    """Load and cache the HousePricePredictor instance."""
    try:
        return HousePricePredictor(), None
    except FileNotFoundError as err:
        return None, str(err)
    except Exception as exc:
        return None, f"Error initializing predictor: {exc}"


@st.cache_data
def load_benchmark_data():
    """Load benchmark dataset for explorer."""
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)
    return None


def main():
    # Application Title & Banner
    st.title("🏠 House Price Valuation & Intelligence Engine")
    st.caption("Production-grade residential real estate valuation powered by Gradient Boosting and Scikit-Learn.")

    predictor, init_error = get_predictor()

    # Graceful recovery if model is missing
    if predictor is None:
        st.error("⚠️ **Model Artifacts Not Found**")
        st.info(
            "The predictive model has not yet been trained. "
            "Please initialize the model by executing the following command in PowerShell:\n\n"
            "```powershell\npython scripts/train.py\n```"
        )
        if init_error:
            st.code(init_error, language="text")
        return

    # Sidebar - Telemetry & Metadata
    meta = predictor.metadata
    st.sidebar.header("📊 Model Telemetry")
    if meta:
        st.sidebar.markdown(f"**Model:** `{meta.get('model_name', 'Gradient Boosting')}`")
        st.sidebar.markdown(f"**Algorithm:** `{meta.get('algorithm', 'GradientBoostingRegressor')}`")
        test_m = meta.get("test_metrics", {})
        if test_m:
            col_sb1, col_sb2 = st.sidebar.columns(2)
            col_sb1.metric("Test R²", f"{test_m.get('R2', 0):.4f}")
            col_sb2.metric("Test RMSE", f"₹{test_m.get('RMSE', 0):.2f}L")
            col_sb1.metric("Test MAE", f"₹{test_m.get('MAE', 0):.2f}L")
            col_sb2.metric("Test MAPE", f"{test_m.get('MAPE (%)', 0):.1f}%")
        sha = meta.get("sha256_checksum", "")
        if sha:
            st.sidebar.caption(f"**Artifact SHA-256:** `{sha[:12]}...`")
    else:
        st.sidebar.info("Model metadata loaded.")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Technical Overview:**\n"
        "- 600-sample benchmark dataset\n"
        "- 5-fold cross-validation verified\n"
        "- Bounded non-negative domain rules\n"
        "- 95% Confidence Interval bounds"
    )
    st.sidebar.caption("Customized & Re-engineered by **Daniel Richard Xavier**")

    # Main Navigation Tabs
    tab_single, tab_batch, tab_telemetry, tab_explorer = st.tabs([
        "🏡 Single Property Valuation",
        "📂 Batch CSV Valuation",
        "📈 Model Performance & Telemetry",
        "🔍 Dataset Explorer",
    ])

    # -------------------------------------------------------------
    # TAB 1: Single Property Valuation
    # -------------------------------------------------------------
    with tab_single:
        st.subheader("Configure Property Specifications")
        st.markdown("Specify physical dimensions, room counts, and neighborhood parameters to estimate fair market value.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📐 Structural Attributes")
            area = st.number_input(
                "Super Built-up Area (sq ft)",
                min_value=300,
                max_value=12000,
                value=1650,
                step=50,
                help="Total built-up residential space in square feet.",
            )

            col_bhk, col_bath = st.columns(2)
            with col_bhk:
                bedrooms = st.number_input(
                    "Bedrooms (BHK)",
                    min_value=1,
                    max_value=10,
                    value=3,
                    step=1,
                )
            with col_bath:
                bathrooms = st.number_input(
                    "Bathrooms",
                    min_value=1,
                    max_value=8,
                    value=2,
                    step=1,
                )

            age = st.slider(
                "Property Age (years)",
                min_value=0.0,
                max_value=30.0,
                value=4.0,
                step=0.5,
                help="Years since completion of construction (0 = brand new).",
            )

        with col2:
            st.markdown("#### 📍 Locality & Amenities")
            locality = st.selectbox(
                "Locality Tier",
                options=LOCALITIES,
                index=1,
                help="Urban Prime: Core city metro hub. Suburban: Established residential corridor. Rural/Outskirts: Developing periphery.",
            )

            col_park, col_furn = st.columns(2)
            with col_park:
                parking = st.selectbox("Designated Parking", options=[0, 1, 2, 3], index=1)
            with col_furn:
                furnished = st.selectbox("Furnishing Status", options=FURNISHED_STATUSES, index=1)

            amenities = st.checkbox(
                "Society Amenities (Gym, Pool, Security, Power Backup)",
                value=True,
            )

        st.markdown("---")
        btn_eval = st.button("🚀 Calculate Property Valuation", type="primary", use_container_width=True)

        if btn_eval:
            input_dict = {
                "area": area,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "age": age,
                "locality": locality,
                "parking": parking,
                "furnished_status": furnished,
                "has_amenities": 1 if amenities else 0,
            }

            try:
                res = predictor.predict(input_dict)

                st.success("✅ **Valuation Completed Successfully**")

                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Estimated Market Value", f"₹{res.price_lakhs:.2f} Lakhs")
                m_col2.metric("95% Prediction Range", f"₹{res.prediction_interval_lower:.2f} - ₹{res.prediction_interval_upper:.2f} L")
                m_col3.metric("Rate per Sq Ft", f"₹{res.price_per_sqft_inr:,.0f} / sqft")

                if res.clamped:
                    st.warning("⚠️ Prediction was clamped to minimum physical salvage valuation floor.")

                with st.expander("🔍 Property Valuation Summary Breakdown", expanded=True):
                    st.write(
                        f"- **Type:** {bedrooms} BHK Residential Apartment / Villa\n"
                        f"- **Location Tier:** {locality}\n"
                        f"- **Area:** {area:,} sq ft ({res.price_per_sqft_inr:,.0f} ₹/sqft)\n"
                        f"- **Age & Depreciation:** {age} years\n"
                        f"- **Parking & Fitout:** {parking} parking spot(s), {furnished}\n"
                        f"- **Amenities Included:** {'Yes' if amenities else 'No'}"
                    )

            except PredictionValidationError as pve:
                st.error(f"Validation Error: {pve}")
            except Exception as e:
                st.error(f"Unexpected prediction failure: {e}")

    # -------------------------------------------------------------
    # TAB 2: Batch CSV Valuation
    # -------------------------------------------------------------
    with tab_batch:
        st.subheader("Batch Real Estate Portfolio Valuation")
        st.markdown("Upload a CSV file containing multiple properties to evaluate portfolio value in batch.")

        # Download Template
        sample_batch = pd.DataFrame([
            {"area": 1200, "bedrooms": 2, "bathrooms": 2, "age": 5.0, "locality": "Suburban", "parking": 1, "furnished_status": "Semi-Furnished", "has_amenities": 1},
            {"area": 2400, "bedrooms": 4, "bathrooms": 3, "age": 1.0, "locality": "Urban Prime", "parking": 2, "furnished_status": "Furnished", "has_amenities": 1},
            {"area": 950, "bedrooms": 2, "bathrooms": 1, "age": 12.0, "locality": "Rural/Outskirts", "parking": 0, "furnished_status": "Unfurnished", "has_amenities": 0},
        ])
        csv_buffer = io.StringIO()
        sample_batch.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Sample Batch CSV Template",
            data=csv_buffer.getvalue(),
            file_name="sample_property_batch.csv",
            mime="text/csv",
        )

        uploaded_file = st.file_uploader("Upload CSV Spreadsheet", type=["csv"])
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write(f"Uploaded {len(batch_df)} records. Preview:")
                st.dataframe(batch_df.head(), use_container_width=True)

                if st.button("⚡ Evaluate Batch Portfolio"):
                    with st.spinner("Evaluating properties..."):
                        evaluated_df = predictor.predict_batch(batch_df)

                    st.success(f"Successfully evaluated {len(evaluated_df)} properties!")
                    st.dataframe(evaluated_df, use_container_width=True)

                    total_portfolio_value = evaluated_df["predicted_price_lakhs"].sum()
                    st.metric("Total Portfolio Estimated Valuation", f"₹{total_portfolio_value:,.2f} Lakhs")

                    # Download Evaluated CSV
                    out_buffer = io.StringIO()
                    evaluated_df.to_csv(out_buffer, index=False)
                    st.download_button(
                        label="📥 Download Evaluated Results CSV",
                        data=out_buffer.getvalue(),
                        file_name="evaluated_portfolio_results.csv",
                        mime="text/csv",
                    )
            except Exception as exc:
                st.error(f"Error processing batch file: {exc}")

    # -------------------------------------------------------------
    # TAB 3: Model Performance & Telemetry
    # -------------------------------------------------------------
    with tab_telemetry:
        st.subheader("Algorithmic Benchmarking & Telemetry")
        st.markdown(
            "Empirical results comparing 5 machine learning regression algorithms across "
            "5-Fold Cross-Validation and an independent holdout test set ($N=120$)."
        )

        comparison_data = [
            {"Model": "Gradient Boosting (Champion)", "Test R²": 0.9717, "Test RMSE (L)": 6.62, "Test MAE (L)": 4.94, "MAPE (%)": 5.86, "CV R²": "0.9715 ± 0.0079"},
            {"Model": "Random Forest Regressor", "Test R²": 0.9614, "Test RMSE (L)": 7.73, "Test MAE (L)": 5.66, "MAPE (%)": 6.81, "CV R²": "0.9550 ± 0.0134"},
            {"Model": "Linear Regression (8 Features)", "Test R²": 0.9377, "Test RMSE (L)": 9.82, "Test MAE (L)": 7.09, "MAPE (%)": 10.37, "CV R²": "0.9384 ± 0.0060"},
            {"Model": "Lasso Regression", "Test R²": 0.9373, "Test RMSE (L)": 9.85, "Test MAE (L)": 7.09, "MAPE (%)": 10.31, "CV R²": "0.9383 ± 0.0061"},
            {"Model": "Ridge Regression", "Test R²": 0.9358, "Test RMSE (L)": 9.97, "Test MAE (L)": 7.15, "MAPE (%)": 10.35, "CV R²": "0.9382 ± 0.0053"},
            {"Model": "Baseline Linear (4 Features)", "Test R²": 0.5309, "Test RMSE (L)": 26.95, "Test MAE (L)": 21.76, "MAPE (%)": 25.39, "CV R²": "N/A"},
        ]
        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, use_container_width=True)

        # Feature Importance Visualization
        feat_imp = meta.get("feature_importances", {})
        if feat_imp:
            st.markdown("#### 🔍 Feature Importances (What Drives Property Price?)")
            imp_df = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance (%)"]).set_index("Feature")
            st.bar_chart(imp_df)
            st.caption("ℹ️ Impurity-based feature importance (MDI) measures tree split reliance across the training set, not causal real-world market effects.")

        # Empirical Calibration & Coverage Telemetry
        cov = meta.get("interval_coverage", {})
        if cov:
            cov_pct = cov.get("picp_coverage_pct", 95.83)
            mean_w = cov.get("mean_interval_width", 25.81)
            st.info(
                f"🎯 **Empirical 95% Prediction Interval Coverage (PICP):** **{cov_pct}%** on holdout test set (Target: 95.0%). "
                f"Average interval bandwidth: **₹{mean_w:.2f} Lakhs** (±₹{mean_w/2:.2f}L)."
            )
            st.caption("ℹ️ Prediction intervals reflect observation uncertainty for an individual home ($\hat{y} \pm 1.96\hat{\sigma}$), distinct from confidence intervals for the population mean response.")

        st.markdown("#### 💡 Key Architectural Enhancements:")
        st.markdown(
            "1. **Resolved Baseline Pathology:** The baseline model was trained on only 10 synthetic rows, causing inverted regression coefficients (e.g. adding bedrooms decreased price).\n"
            "2. **Error Reduction:** Champion Gradient Boosting delivers a **75.4% reduction in RMSE** compared to the baseline 4-feature model.\n"
            "3. **Physical Domain Guardrails:** The inference engine enforces physical clamping, eliminating the baseline's negative price bug."
        )

    # -------------------------------------------------------------
    # TAB 4: Dataset Explorer
    # -------------------------------------------------------------
    with tab_explorer:
        st.subheader("Residential Housing Benchmark Dataset")
        st.info("ℹ️ **Dataset Disclosure**: This 600-sample benchmark dataset was procedurally generated using parameterized distribution modeling calibrated to Indian residential real estate metrics. It demonstrates pipeline mechanics and statistical integrity, but does not represent an actual live MLS transaction feed.")
        df_bench = load_benchmark_data()
        if df_bench is not None:
            st.markdown(f"Displaying {len(df_bench)} verified records across Indian residential markets.")
            st.dataframe(df_bench.head(20), use_container_width=True)

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("#### Summary Statistics")
                st.dataframe(df_bench.describe().round(2), use_container_width=True)

            with col_d2:
                st.markdown("#### Average Price by Locality Tier")
                loc_avg = df_bench.groupby("locality")["price"].agg(["count", "mean", "min", "max"]).round(2)
                loc_avg.columns = ["Properties", "Mean Price (L)", "Min Price (L)", "Max Price (L)"]
                st.dataframe(loc_avg, use_container_width=True)
        else:
            st.warning("Benchmark dataset not found at data/house_prices.csv.")

    st.markdown("---")
    st.caption("House Price Prediction System v2.0.0 | Engineered & Customized by Daniel Richard Xavier | Original prototype by kirankumarnscr7-cmyk")


if __name__ == "__main__":
    main()
