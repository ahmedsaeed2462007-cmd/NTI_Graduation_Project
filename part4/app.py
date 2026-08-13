import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="SolarPath - Intelligent Solar Analytics Platform",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .status-normal {
        color: #059669;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .status-anomaly {
        color: #DC2626;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load models safely
@st.cache_resource
def load_models():
    models = {}
    try:
        # Phase 1
        models['pt_clim'] = joblib.load('phase1_power_transformer.joblib')
        models['pca_clim'] = joblib.load('phase1_pca.joblib')
        models['km_clim'] = joblib.load('phase1_kmeans.joblib')
        # Phase 2
        models['stacking'] = joblib.load('solarpath_stacking_model.joblib')
        models['features_solar'] = joblib.load('solarpath_features_list.joblib')
        # Phase 3
        models['scaler_anomaly'] = joblib.load('solarpath_anomaly_scaler.joblib')
        models['iso_forest'] = joblib.load('solarpath_iso_forest.joblib')
    except Exception as e:
        st.error(f"Error loading model files: {e}. Please ensure all .joblib files are in the repository.")
    return models

models = load_models()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/solar-panel.png", width=80)
st.sidebar.title("SolarPath Platform")
st.sidebar.write("National Telecommunication Institute (NTI)")
st.sidebar.write("---")

app_mode = st.sidebar.selectbox(
    "Choose Phase Analysis:",
    [
        "🌍 Phase 1: Site Suitability (Egypt)",
        "⚡ Phase 2: Performance & Power Predictor",
        "🔍 Phase 3: Operational Fault Detector"
    ]
)

st.sidebar.write("---")
st.sidebar.markdown("""
### Model Architecture Info
*   **Clustering**: Yeo-Johnson + PCA + K-Means ($K=2$)
*   **Regression**: Stacking Regressor (RF + XGBoost + GBM $\\to$ Ridge)
*   **Anomaly**: Unsupervised Isolation Forest (5D space)
""")

# ==========================================
# PHASE 1: SITE SUITABILITY (EGYPT)
# ==========================================
if app_mode == "🌍 Phase 1: Site Suitability (Egypt)":
    st.markdown('<div class="main-header">🌍 Egypt Climatology & Site Suitability</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Identify optimal geographical coordinates across Egypt for solar energy development based on NASA POWER annual climate features.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Enter Annual average climate values:")
        solar = st.slider("Annual Average Solar Irradiance (kWh/m²/day)", 4.0, 8.0, 6.5, 0.1)
        humidity = st.slider("Annual Average Relative Humidity (%)", 10.0, 90.0, 30.0, 1.0)
        temp = st.slider("Annual Average Air Temperature (°C)", 10.0, 40.0, 24.0, 0.5)
        wind = st.slider("Annual Average Wind Speed (m/s)", 1.0, 10.0, 3.0, 0.1)

        submitted = st.button("Analyze Site Suitability", type="primary")

    with col2:
        if submitted and 'km_clim' in models:
            # Prepare input
            input_data = pd.DataFrame([[solar, humidity, temp, wind]], columns=['ALLSKY_SFC_SW_DWN_ANN', 'RH2M_ANN', 'T2M_ANN', 'WS2M_ANN'])
            
            # Apply transforms
            X_trans = models['pt_clim'].transform(input_data)
            X_pca = models['pca_clim'].transform(X_trans)
            
            # Predict
            cluster = models['km_clim'].predict(X_pca)[0]
            
            st.subheader("Suitability Assessment Results:")
            if cluster == 0:
                st.success("### ★ HIGHLY RECOMMENDED SITE (Cluster 0) ★")
                st.write("""
                **Reasoning**: This geographical zone belongs to the **Optimal Egypt Solar Belt (Upper Egypt & Southern Deserts)**. 
                *   **Advantages**: Maximized solar radiation intensity, minimal cloud coverage, clear desert skies, and low humidity.
                *   **Estimated Annual Yield**: Excellent ($> 6.4$ kWh/m²/day).
                """)
            else:
                st.warning("### NOT RECOMMENDED FOR MACRO DEVELOPMENTS (Cluster 1)")
                st.write("""
                **Reasoning**: This geographical zone belongs to **Coastal/Nile Delta regions**.
                *   **Disadvantages**: Moderately lower solar radiation, higher humidity (which reduces panel efficiency), and higher frequency of cloud coverage.
                *   **Suitability**: Ideal for residential micro-generation, but not optimal for large-scale utility solar farms.
                """)
            
            # Comparison table
            st.write("---")
            st.write("#### Comparison with Cluster Averages:")
            comparison_df = pd.DataFrame({
                "Feature": ["Solar Irradiance (kWh/m²/d)", "Relative Humidity (%)", "Temperature (°C)", "Wind Speed (m/s)"],
                "Your Input": [solar, humidity, temp, wind],
                "Cluster 0 (Best) Average": [6.526, 28.83, 24.12, 3.14],
                "Cluster 1 Average": [6.108, 44.91, 21.25, 3.13]
            })
            st.table(comparison_df.set_index("Feature"))
        else:
            st.info("Adjust the sliders on the left and click 'Analyze Site Suitability' to view results.")

# ==========================================
# PHASE 2: PERFORMANCE & POWER PREDICTOR
# ==========================================
elif app_mode == "⚡ Phase 2: Performance & Power Predictor":
    st.markdown('<div class="main-header">⚡ Solar Performance & Power Generation Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict expected solar module Performance Ratio (PR) and total power output using weather readings.</div>', unsafe_allow_html=True)

    mode = st.radio("Select Prediction Mode:", ["Manual Entry", "Sequence Upload (Batch Mode)"])

    if mode == "Manual Entry":
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("1. Enter Current Weather & Module Readings:")
            temp_ambient = st.number_input("Ambient Temperature (°C)", 0.0, 55.0, 30.0, 0.5)
            temp_module = st.number_input("Module Temperature (°C)", 0.0, 85.0, 45.0, 0.5)
            humidity = st.number_input("Relative Humidity (%)", 5.0, 100.0, 40.0, 1.0)
            wind_speed = st.number_input("Wind Speed (m/s)", 0.0, 25.0, 3.5, 0.1)
            irradiation = st.number_input("Solar Irradiance (kW/m²)", 0.0, 1.5, 0.8, 0.05)

            st.subheader("2. Solar Plant Parameters:")
            area = st.number_input("Total Panel Surface Area (m²)", 1.0, 10000000.0, 960000.0, 100.0)
            efficiency = st.number_input("Solar Module Efficiency (%)", 5.0, 25.0, 17.0, 0.5) / 100.0

            predict_btn = st.button("Predict Expected Output", type="primary")

        with col2:
            if predict_btn and 'stacking' in models:
                # Prepare single-row lag features (for manual single input, history is assumed steady)
                # Features list expected: ['temp_ambient', 'temp_module', 'wind_speed', 'humidity', 
                #                          'temp_module_lag1', 'temp_module_lag2', 'temp_module_roll_mean_3', 'temp_module_roll_std_3', ...]
                temp_delta = temp_module - temp_ambient
                
                input_dict = {
                    'temp_ambient': temp_ambient,
                    'temp_module': temp_module,
                    'wind_speed': wind_speed,
                    'humidity': humidity,
                    # Module temp history
                    'temp_module_lag1': temp_module,
                    'temp_module_lag2': temp_module,
                    'temp_module_roll_mean_3': temp_module,
                    'temp_module_roll_std_3': 0.0,
                    # Ambient temp history
                    'temp_ambient_lag1': temp_ambient,
                    'temp_ambient_lag2': temp_ambient,
                    'temp_ambient_roll_mean_3': temp_ambient,
                    'temp_ambient_roll_std_3': 0.0,
                    # Humidity history
                    'humidity_lag1': humidity,
                    'humidity_lag2': humidity,
                    'humidity_roll_mean_3': humidity,
                    'humidity_roll_std_3': 0.0,
                    # Wind history
                    'wind_speed_lag1': wind_speed,
                    'wind_speed_lag2': wind_speed,
                    'wind_speed_roll_mean_3': wind_speed,
                    'wind_speed_roll_std_3': 0.0,
                    # Delta
                    'temp_delta': temp_delta
                }
                
                input_df = pd.DataFrame([input_dict])
                
                # Align columns order exactly with training features
                X_input = input_df[models['features_solar']]
                
                # Predict
                expected_pr = models['stacking'].predict(X_input)[0]
                expected_pr = np.clip(expected_pr, 0.0, 1.2)
                
                # Nighttime check
                if irradiation < 0.01:
                    expected_pr = 0.0
                
                # Physical formulas
                theoretical_power = irradiation * area * efficiency
                predicted_power = expected_pr * theoretical_power
                
                # Show Metrics
                st.subheader("Prediction Results:")
                st.write("---")
                
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Predicted Performance Ratio (PR)", f"{expected_pr:.3f}")
                with m2:
                    st.metric("Estimated Active Power Output (kW)", f"{predicted_power:,.2f}")
                
                st.write("---")
                st.write("#### Detailed Calculations:")
                st.write(f"- **Theoretical Power (100% capacity)**: `{theoretical_power:,.2f} kW` (based on area & efficiency)")
                st.write(f"- **Expected Power (After Losses)**: `{predicted_power:,.2f} kW` (based on PR of `{expected_pr*100:.1f}%`)")
                
                # Store prediction for next tab
                st.session_state['expected_pr'] = expected_pr
                st.session_state['irradiation'] = irradiation
                st.session_state['temp_delta'] = temp_delta
            else:
                st.info("Input weather details and click 'Predict Expected Output' to calculate performance and power production.")

    elif mode == "Sequence Upload (Batch Mode)":
        st.subheader("Upload Time Series Sequence (CSV):")
        st.write("Upload a CSV file containing sequential columns: `timestamp`, `temp_ambient`, `temp_module`, `humidity`, `wind_speed`, `irradiation`.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None and 'stacking' in models:
            batch_df = pd.read_csv(uploaded_file)
            st.success("CSV Uploaded successfully!")
            
            # Verify columns
            required_cols = ['temp_ambient', 'temp_module', 'humidity', 'wind_speed', 'irradiation']
            if not all(col in batch_df.columns for col in required_cols):
                st.error(f"Missing required columns! File must contain: {required_cols}")
            else:
                # Process Lags and Rollings
                df_proc = batch_df.copy()
                df_proc['timestamp'] = pd.to_datetime(df_proc.get('timestamp', pd.date_range(start='2020-01-01', periods=len(df_proc), freq='15min')))
                df_proc = df_proc.sort_values('timestamp').reset_index(drop=True)
                
                # Compute sequential lags
                for col in ['temp_module', 'temp_ambient', 'humidity', 'wind_speed']:
                    df_proc[f'{col}_lag1'] = df_proc[col].shift(1).ffill().bfill()
                    df_proc[f'{col}_lag2'] = df_proc[col].shift(2).ffill().bfill()
                    df_proc[f'{col}_roll_mean_3'] = df_proc[col].rolling(window=3, min_periods=1).mean()
                    df_proc[f'{col}_roll_std_3'] = df_proc[col].rolling(window=3, min_periods=1).std().fillna(0)
                
                df_proc['temp_delta'] = df_proc['temp_module'] - df_proc['temp_ambient']
                
                # Align features
                X_batch = df_proc[models['features_solar']]
                
                # Predict
                preds = models['stacking'].predict(X_batch)
                preds = np.clip(preds, 0.0, 1.2)
                
                # Nighttime masking
                preds = np.where(df_proc['irradiation'] < 0.01, 0.0, preds)
                
                df_proc['Expected_PR'] = preds
                
                # Estimated Power
                area = st.number_input("Batch Surface Area (m²)", 1.0, 10000000.0, 960000.0, 100.0)
                efficiency = st.number_input("Batch Module Efficiency (%)", 5.0, 25.0, 17.0, 0.5) / 100.0
                
                df_proc['Predicted_Power_kW'] = df_proc['Expected_PR'] * df_proc['irradiation'] * area * efficiency
                
                # Visualizations
                st.subheader("Batch Prediction Outputs:")
                st.write(df_proc[['timestamp', 'irradiation', 'temp_ambient', 'Expected_PR', 'Predicted_Power_kW']].head(10))
                
                st.line_chart(data=df_proc.set_index('timestamp')['Predicted_Power_kW'], use_container_width=True)
                
                # Download button
                csv_data = df_proc.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Processed Predictions CSV",
                    data=csv_data,
                    file_name="solarpath_batch_predictions.csv",
                    mime="text/csv"
                )

# ==========================================
# PHASE 3: OPERATIONAL FAULT DETECTOR
# ==========================================
elif app_mode == "🔍 Phase 3: Operational Fault Detector":
    st.markdown('<div class="main-header">🔍 Operational Anomaly & Fault Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unsupervised machine learning fault detection using Isolation Forests to capture sub-optimal panel degradation.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Operational Readings:")
        
        # Auto-fill from Tab 2 if exists
        default_exp = st.session_state.get('expected_pr', 0.8)
        default_irr = st.session_state.get('irradiation', 0.8)
        default_delta = st.session_state.get('temp_delta', 15.0)
        
        expected_pr = st.number_input("Expected Performance Ratio (from weather model)", 0.0, 1.2, default_exp, 0.01)
        actual_pr = st.number_input("Actual Observed Performance Ratio (from real meters)", 0.0, 1.2, 0.65, 0.01)
        irradiation = st.number_input("Current Solar Irradiance (kW/m²)", 0.0, 1.5, default_irr, 0.05)
        temp_delta = st.number_input("Thermal Delta (temp_module - temp_ambient) (°C)", -10.0, 50.0, default_delta, 0.5)

        run_btn = st.button("Run Fault Analysis", type="primary")

    with col2:
        if run_btn and 'iso_forest' in models:
            deviation = actual_pr - expected_pr
            
            # Anomaly features order: ['Actual_Ratio', 'Expected_Ratio', 'Deviation', 'irradiation', 'temp_delta']
            input_df = pd.DataFrame([[actual_pr, expected_pr, deviation, irradiation, temp_delta]], 
                                    columns=['Actual_Ratio', 'Expected_Ratio', 'Deviation', 'irradiation', 'temp_delta'])
            
            # Scale
            X_scaled = models['scaler_anomaly'].transform(input_df)
            
            # Predict
            pred = models['iso_forest'].predict(X_scaled)[0]
            anomaly_score = -models['iso_forest'].decision_function(X_scaled)[0]
            
            st.subheader("Fault Detection Results:")
            st.write("---")
            
            # Display results
            if pred == 1:
                st.markdown('<span class="status-normal">🟢 NORMAL OPERATION (No Faults Detected)</span>', unsafe_allow_html=True)
                st.write(f"The inverter performance aligns with local weather conditions. Deviation is stable at `{deviation:+.3f}`.")
            else:
                st.markdown('<span class="status-anomaly">🚨 ANOMALY DETECTED (Potential Equipment Fault / Degradation)</span>', unsafe_allow_html=True)
                st.write(f"**Anomaly Score**: `{anomaly_score:.3f}` | **Performance Deviation**: `{deviation:+.3f}`")
                
                # Analyze deviation direction
                if deviation < -0.10:
                    st.error("""
                    **Diagnosis**: Significant Under-performance.
                    *   **Possible Causes**: Extreme dust/soiling accumulation, severe cloud shadows, panels degradation, shading from nearby structures, or individual cell wiring issues (string faults).
                    *   **Recommendation**: Dispatch maintenance to inspect and clean panel surfaces.
                    """)
                elif deviation > 0.15:
                    st.info("""
                    **Diagnosis**: Unusually high performance.
                    *   **Possible Causes**: Sensor calibration issues (irradiance reading is too low) or model boundary misalignment.
                    """)
                else:
                    st.warning("""
                    **Diagnosis**: Moderate deviation anomaly.
                    *   **Possible Causes**: Intermittent grid matching delays, brief shading, or mild inverter clipping.
                    """)
            
            st.write("---")
            st.write("#### Multi-Dimensional Feature Diagnostics:")
            st.write(f"- **Deviation (Actual - Expected)**: `{deviation:+.3f}`")
            st.write(f"- **Solar Irradiance**: `{irradiation:.3f} kW/m²`")
            st.write(f"- **Thermal Delta (heating level)**: `{temp_delta:.2f}°C`")
        else:
            st.info("Input operational readings and click 'Run Fault Analysis' to test for system failures.")
