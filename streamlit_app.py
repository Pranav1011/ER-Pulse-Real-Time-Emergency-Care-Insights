import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Healthcare Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
)
st.title("🏥 Healthcare Analytics Dashboard")

# Create tabs
tab_dashboard, tab_models, tab_explain = st.tabs([
    "🏥 Dashboard",
    "📊 Model Comparison",
    "🔍 Explainability"
])

# Load model and encoder
model_path = 'src/models/ed_wait_time_model.pkl'
encoder_path = 'src/models/encoder.pkl'

try:
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    model_loaded = True
except Exception as e:
    model_loaded = False


@st.cache_data
def load_data():
    admissions = pd.read_csv('data/processed/admissions.csv', parse_dates=['admittime', 'dischtime', 'edregtime', 'edouttime'])
    hourly_stats_admissions = pd.read_csv('data/processed/hourly_stats_admissions.csv')
    hourly_stats_transfers = pd.read_csv('data/processed/hourly_stats_transfers.csv')
    metrics_wait_times = pd.read_csv('data/processed/metrics_wait_times.csv')
    ward_metrics = pd.read_csv('data/processed/metrics_ward_metrics.csv')
    return admissions, hourly_stats_admissions, hourly_stats_transfers, metrics_wait_times, ward_metrics


@st.cache_data
def load_model_comparison_results():
    """Load model comparison results if available."""
    results_path = Path('src/models/model_comparison_results.csv')
    if results_path.exists():
        return pd.read_csv(results_path)
    return None


@st.cache_data
def load_feature_importance():
    """Load SHAP feature importance if available."""
    importance_path = Path('src/models/explanations/feature_importance.csv')
    if importance_path.exists():
        return pd.read_csv(importance_path)
    return None


admissions, hourly_admissions, hourly_transfers, wait_times_metrics, ward_metrics = load_data()

# ==================== DASHBOARD TAB ====================
with tab_dashboard:
    st.sidebar.title("⚙️ Dashboard Controls")

    # Hourly Admission Patterns
    st.header("📊 Hourly Admission Patterns")
    admission_type = st.sidebar.selectbox(
        "Select Admission Type",
        options=hourly_admissions['admission_type'].unique(),
        index=0
    )
    hour_range = st.sidebar.slider(
        "Select Hour Range",
        min_value=0,
        max_value=23,
        value=(0, 23),
        step=1
    )
    filtered_admissions = hourly_admissions[
        (hourly_admissions['admission_type'] == admission_type) &
        (hourly_admissions['hour'] >= hour_range[0]) &
        (hourly_admissions['hour'] <= hour_range[1])
    ]
    fig_admissions = px.area(
        filtered_admissions,
        x='hour',
        y='admission_count',
        title=f"Hourly Admissions for {admission_type}",
        labels={'hour': 'Hour of Day', 'admission_count': 'Number of Admissions'},
        color_discrete_sequence=['#1f77b4']
    )
    st.plotly_chart(fig_admissions, use_container_width=True)

    # Two column layout
    col1, col2 = st.columns(2)

    with col1:
        # Admission Type Distribution Pie Chart
        st.header("**Admission Type Distribution**")
        admission_distribution = hourly_admissions.groupby('admission_type')['admission_count'].sum().reset_index()
        fig_pie = px.pie(
            admission_distribution,
            names='admission_type',
            values='admission_count',
            title="Admission Type Distribution",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Department Utilization
        st.header("🏥 Department Utilization")
        selected_ward = st.sidebar.selectbox(
            "Select Ward ID",
            options=ward_metrics['curr_wardid'].unique()
        )
        filtered_ward = ward_metrics[ward_metrics['curr_wardid'] == selected_ward]
        if not filtered_ward.empty:
            st.write(f"Metrics for Ward ID: {selected_ward}")
            st.metric("Average Length of Stay (hrs)", f"{filtered_ward['length_of_stay_mean'].values[0]:.1f}")
            st.metric("Median Length of Stay (hrs)", f"{filtered_ward['length_of_stay_median'].values[0]:.1f}")
            st.metric("Patient Count", int(filtered_ward['subject_id_count'].values[0]))
        else:
            st.warning(f"No data available for Ward ID {selected_ward}")

    # Wait Time Analysis
    st.header("⏱️ Wait Time Analysis")
    wait_type = st.sidebar.selectbox(
        "Select Admission Type for Wait Time Analysis",
        options=wait_times_metrics.index.unique()
    )
    filtered_wait_time = wait_times_metrics.loc[wait_type]
    col_wait1, col_wait2 = st.columns(2)
    col_wait1.metric("Mean ED Wait Time (min)", f"{filtered_wait_time['ed_wait_time_mean']:.1f}")
    col_wait2.metric("Median ED Wait Time (min)", f"{filtered_wait_time['ed_wait_time_median']:.1f}")

    # Prediction Section
    st.header("🔮 Predict ED Wait Times")
    if model_loaded:
        col_pred1, col_pred2 = st.columns(2)

        with col_pred1:
            admission_type_input = st.selectbox("Admission Type", options=['Emergency', 'Elective', 'Urgent'])
            hour_input = st.slider("Hour of Day", 0, 23, value=12)

        with col_pred2:
            admission_location_input = st.text_input("Admission Location", "CLINIC")
            ethnicity_input = st.text_input("Ethnicity", "WHITE")

        input_data = pd.DataFrame({
            'admission_type': [admission_type_input],
            'hour': [hour_input],
            'admission_location': [admission_location_input],
            'ethnicity': [ethnicity_input]
        })

        if st.button("Predict Wait Time"):
            try:
                input_encoded = encoder.transform(input_data)
                predicted_wait_time = model.predict(input_encoded)
                st.success(f"Predicted ED Wait Time: **{predicted_wait_time[0]:.2f} minutes**")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
    else:
        st.warning("Model not loaded. Please train and save the model first.")

    # Transfer Patterns
    st.header("🔄 Transfer Patterns")
    transfer_hour_range = st.sidebar.slider(
        "Select Hour Range for Transfers",
        min_value=0,
        max_value=23,
        value=(0, 23),
        step=1
    )
    filtered_transfers = hourly_transfers[
        (hourly_transfers['hour'] >= transfer_hour_range[0]) &
        (hourly_transfers['hour'] <= transfer_hour_range[1])
    ]
    fig_transfers = px.bar(
        filtered_transfers,
        x='hour',
        y='transfer_count',
        color='curr_wardid',
        title="Hourly Transfer Patterns",
        labels={'hour': 'Hour of Day', 'transfer_count': 'Transfer Count', 'curr_wardid': 'Ward ID'}
    )
    st.plotly_chart(fig_transfers, use_container_width=True)


# ==================== MODEL COMPARISON TAB ====================
with tab_models:
    st.header("📊 Model Comparison Results")
    st.markdown("""
    Compare multiple machine learning models for ED wait time prediction.
    Models are evaluated using cross-validation with statistical significance tests.
    """)

    # Load comparison results
    comparison_df = load_model_comparison_results()

    if comparison_df is not None:
        # Metrics overview
        st.subheader("Model Performance Metrics")

        # Bar chart of MAE
        fig_mae = px.bar(
            comparison_df,
            x='Model',
            y='MAE',
            error_y='CV MAE (std)',
            title='Mean Absolute Error by Model',
            color='MAE',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig_mae, use_container_width=True)

        # R² comparison
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            fig_r2 = px.bar(
                comparison_df,
                x='Model',
                y='R²',
                title='R² Score by Model',
                color='R²',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_r2, use_container_width=True)

        with col_m2:
            fig_time = px.bar(
                comparison_df,
                x='Model',
                y='Training Time (s)',
                title='Training Time by Model',
                color='Training Time (s)',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_time, use_container_width=True)

        # Full results table
        st.subheader("Detailed Results")
        st.dataframe(comparison_df, use_container_width=True)

        # Best model highlight
        best_model = comparison_df.loc[comparison_df['CV MAE (mean)'].idxmin()]
        st.success(f"**Best Model:** {best_model['Model']} (CV MAE: {best_model['CV MAE (mean)']:.2f})")

        # Statistical analysis report
        report_path = Path('src/models/statistical_report.md')
        if report_path.exists():
            with st.expander("📊 Statistical Analysis Report"):
                st.markdown(report_path.read_text())

    else:
        st.info("No model comparison results found. Run the model comparison script to generate results.")

        st.code("""
# Run model comparison:
from src.predictive_modeling.model_comparison import run_comparison
results = run_comparison()
        """, language='python')


# ==================== EXPLAINABILITY TAB ====================
with tab_explain:
    st.header("🔍 Model Explainability")
    st.markdown("""
    Understand how the model makes predictions using SHAP (SHapley Additive exPlanations).
    SHAP values show the contribution of each feature to individual predictions.
    """)

    # Load feature importance
    importance_df = load_feature_importance()

    if importance_df is not None:
        # Global feature importance
        st.subheader("Global Feature Importance")

        fig_importance = px.bar(
            importance_df.head(20),
            x='importance',
            y='feature',
            orientation='h',
            title='Top 20 Features by SHAP Importance',
            labels={'importance': 'Mean |SHAP Value|', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_importance, use_container_width=True)

        # Feature importance table
        with st.expander("📋 All Feature Importances"):
            st.dataframe(importance_df, use_container_width=True)

        # Load and display report
        report_path = Path('src/models/explanations/explainability_report.md')
        if report_path.exists():
            with st.expander("📄 Full Explainability Report"):
                st.markdown(report_path.read_text())

        # Individual prediction explanation
        st.subheader("Explain Individual Prediction")

        if model_loaded:
            col_e1, col_e2 = st.columns(2)

            with col_e1:
                exp_admission_type = st.selectbox("Admission Type (Explain)", options=['Emergency', 'Elective', 'Urgent'], key='exp_type')
                exp_hour = st.slider("Hour of Day (Explain)", 0, 23, value=12, key='exp_hour')

            with col_e2:
                exp_location = st.text_input("Admission Location (Explain)", "CLINIC", key='exp_loc')
                exp_ethnicity = st.text_input("Ethnicity (Explain)", "WHITE", key='exp_eth')

            if st.button("Explain Prediction"):
                try:
                    # Prepare input
                    input_data = pd.DataFrame({
                        'admission_type': [exp_admission_type],
                        'hour': [exp_hour],
                        'admission_location': [exp_location],
                        'ethnicity': [exp_ethnicity]
                    })
                    input_encoded = encoder.transform(input_data)

                    # Get prediction
                    prediction = model.predict(input_encoded)[0]
                    st.info(f"**Predicted Wait Time:** {prediction:.2f} minutes")

                    # Try to get SHAP explanation
                    try:
                        import shap
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(input_encoded)

                        # Create explanation DataFrame
                        feature_names = encoder.get_feature_names_out(['admission_type', 'hour', 'admission_location', 'ethnicity'])
                        explanation_df = pd.DataFrame({
                            'feature': feature_names,
                            'shap_value': shap_values[0]
                        })
                        explanation_df['abs_shap'] = np.abs(explanation_df['shap_value'])
                        explanation_df = explanation_df.sort_values('abs_shap', ascending=False)

                        # Show top contributors
                        st.write("**Top Contributing Features:**")
                        top_features = explanation_df.head(10)

                        fig_explain = px.bar(
                            top_features,
                            x='shap_value',
                            y='feature',
                            orientation='h',
                            title='Feature Contributions to This Prediction',
                            labels={'shap_value': 'SHAP Value (minutes)', 'feature': 'Feature'},
                            color='shap_value',
                            color_continuous_scale='RdBu',
                            color_continuous_midpoint=0
                        )
                        fig_explain.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig_explain, use_container_width=True)

                    except Exception as e:
                        st.warning(f"Could not generate SHAP explanation: {e}")

                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Model not loaded. Cannot generate explanations.")

    else:
        st.info("No explainability data found. Run the explainability script to generate SHAP values.")

        st.code("""
# Generate SHAP explanations:
from src.predictive_modeling.explainability import explain_model
results = explain_model()
        """, language='python')


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>"
    "<b>Created by Sai Pranav</b><br>"
    "Powered by Streamlit | Data Source: MIMIC-III</div>",
    unsafe_allow_html=True
)
