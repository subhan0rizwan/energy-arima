# app.py (or streamlit_app.py)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import os
import pickle
import torch
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import importlib.util

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Energy Price Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a Vibrant, Modern Theme ---
st.markdown("""
    <style>
    /* Main body and background */
    .stApp {
        background-color: #1c1c1c; /* Dark charcoal background */
        color: #f0f2f6;
    }

    /* Buttons */
    .stButton>button {
        background-color: #00e6e6; /* Electric blue */
        color: #1c1c1c;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        box-shadow: 0 4px 6px rgba(0, 230, 230, 0.3);
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #00ff66; /* Energetic green on hover */
        color: #1c1c1c;
        transform: translateY(-2px);
        box-shadow: 0 6px 10px rgba(0, 255, 102, 0.4);
    }

    /* Selectbox and Other Widgets */
    .stSelectbox, .stSlider {
        background-color: #2c2c2c;
        border-radius: 8px;
        border: 1px solid #3c3c3c;
    }

    /* Titles and Headers */
    h1 {
        color: #ffffff;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5);
        border-bottom: 2px solid #00e6e6; /* Title underline matches button color */
        padding-bottom: 10px;
    }
    h3 {
        color: #ffffff;
        margin-top: 30px;
    }

    /* Charts and DataFrames */
    .stPlotlyChart {
        box-shadow: 8px 8px 15px rgba(0, 0, 0, 0.6);
        border-radius: 12px;
        overflow: hidden; /* Ensures the shadow respects the border radius */
    }
    
    /* Animation for a smooth entrance */
    .animate__fadeIn {
        animation: fadeIn 1.5s ease-in;
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# --- Dependency Check ---
required_libraries = ['pandas', 'numpy', 'yfinance', 'plotly', 'torch', 'statsmodels', 'arch', 'prophet', 'sklearn', 'transformers']
missing_libraries = [lib for lib in required_libraries if not importlib.util.find_spec(lib)]
if missing_libraries:
    st.error(f"Missing critical libraries: {', '.join(missing_libraries)}. Please run 'pip install -r requirements.txt'.")
    st.stop()

# --- Utility Functions ---
def preprocess_for_prediction(df):
    """
    A lightweight version of the main preprocessing script, designed to prepare
    data for generating new predictions without re-running the full analysis.
    """
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    base_cols = [col for col in df.columns if '_rolling' not in col and '_std' not in col]
    for col in base_cols:
        if col in ['Petrol', 'Uranium', 'Natural_Gas', 'Crude_Oil']:
            df[f'{col}_rolling'] = df[col].rolling(12).mean()
            df[f'{col}_std'] = df[col].rolling(12).std()
            
    df = df.ffill().bfill()
    df.columns = df.columns.astype(str)
    # Add placeholder for sentiment
    if 'Sentiment' not in df.columns:
        df['Sentiment'] = 0.0
    return df

# Defining the LSTM model class is necessary to load the saved state_dict.
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=3, dropout_rate=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- Data and Model Loading ---
@st.cache_data
def load_data():
    try:
        df = pd.read_parquet('energy_prices.parquet')
        results = pd.read_csv('model_results.csv')
        return df, results
    except FileNotFoundError:
        st.error("Data and model files not found. Please run 'python main.py' to generate them.")
        st.stop()

@st.cache_resource
def load_models(input_size):
    models = {}
    try:
        with open('arima_model.pkl', 'rb') as f: models['arima'] = pickle.load(f)
    except FileNotFoundError: models['arima'] = None
    try:
        with open('prophet_model.pkl', 'rb') as f: models['prophet'] = pickle.load(f)
    except FileNotFoundError: models['prophet'] = None
    try:
        model = LSTMModel(input_size=input_size)
        model.load_state_dict(torch.load('lstm_model.pth'))
        model.eval()
        models['lstm'] = model
    except (FileNotFoundError, RuntimeError) as e:
        st.warning(f"LSTM model could not be loaded: {e}. Please ensure it was trained successfully.")
        models['lstm'] = None
    return models

# --- Main App Logic ---
required_files = ['energy_prices.parquet', 'arima_model.pkl', 'lstm_model.pth', 'prophet_model.pkl', 'model_results.csv']
if not all(os.path.exists(f) for f in required_files):
    st.warning("Required data and model files are missing. Please generate them by running `python main.py`.")
    st.stop()

df, results = load_data()

processed_df_for_size = preprocess_for_prediction(df.copy())
models = load_models(input_size=len(processed_df_for_size.columns))
energy_types = [col for col in df.columns if col in ['Petrol', 'Uranium', 'Natural_Gas', 'Crude_Oil']]

# --- Sidebar Controls ---
st.sidebar.title("⚙️ Controls")
selected_energy = st.sidebar.selectbox("Select Energy Type", options=energy_types)
forecast_horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    options=[3, 6, 12, 24],
    format_func=lambda x: f"{x} Months",
    index=1
)

# --- Dashboard Layout ---
st.markdown('<h1 class="animate__fadeIn">Subhan Rizwan - Energy Price Forecasting Dashboard</h1>', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Data Overview", "Forecasts", "Advanced Analyses"])

with tab1:
    st.markdown('<h3 class="animate__fadeIn">Historical Prices</h3>', unsafe_allow_html=True)
    fig_price = px.line(df, x=df.index, y=selected_energy, title=f'{selected_energy} Historical Prices')
    
    fig_price.update_traces(line_color='#00e6e6') # Set line color
    fig_price.update_layout(
        plot_bgcolor='#2c2c2c', paper_bgcolor='#2c2c2c', font_color='#ffffff',
        template='plotly_dark', hovermode='x unified',
        title_font_color='white'
    )
    
    st.plotly_chart(fig_price, use_container_width=True)
    st.dataframe(df.reset_index().rename(columns={'index':'Date'}), use_container_width=True)

with tab2:
    st.markdown('<h3 class="animate__fadeIn">Model Forecasts</h3>', unsafe_allow_html=True)
    
    last_date = df.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_horizon, freq='M')
    
    full_processed_df = preprocess_for_prediction(df.copy())

    # --- Generate Forecasts ---
    arima_preds, lstm_preds, prophet_preds = None, None, None
    
    if models.get('arima'):
        try:
            exog_features = [col for col in full_processed_df.columns if col != selected_energy and ('rolling' in col or 'std' in col or 'Sentiment' in col)]
            last_exog = full_processed_df[exog_features].iloc[-1:].values
            future_exog = np.repeat(last_exog, forecast_horizon, axis=0)
            arima_preds = models['arima'].forecast(steps=forecast_horizon, exog=future_exog)
        except Exception as e:
            st.warning(f"ARIMA prediction failed: {e}")

    if models.get('prophet'):
        try:
            future_df = pd.DataFrame({'ds': forecast_dates})
            for regressor in models['prophet'].extra_regressors:
                future_df[regressor] = full_processed_df[regressor].iloc[-1]
            prophet_preds = models['prophet'].predict(future_df)['yhat']
        except Exception as e: st.warning(f"Prophet prediction failed: {e}")
    
    if models.get('lstm'):
        try:
            seq_length = 12
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_df = pd.DataFrame(scaler.fit_transform(full_processed_df), columns=full_processed_df.columns)
            
            last_sequence = torch.tensor(scaled_df.values[-seq_length:], dtype=torch.float32).unsqueeze(0)
            preds_scaled_list = []
            for _ in range(forecast_horizon):
                with torch.no_grad():
                    pred = models['lstm'](last_sequence)
                    preds_scaled_list.append(pred.item())
                    new_row = last_sequence.squeeze(0)[-1].numpy()
                    target_idx = scaled_df.columns.get_loc(selected_energy)
                    new_row[target_idx] = pred.item()
                    last_sequence = torch.cat((last_sequence.squeeze(0)[1:], torch.tensor(new_row).unsqueeze(0)), 0).unsqueeze(0)

            target_idx = full_processed_df.columns.get_loc(selected_energy)
            dummy_preds = np.zeros((forecast_horizon, len(full_processed_df.columns)))
            dummy_preds[:, target_idx] = preds_scaled_list
            lstm_preds = scaler.inverse_transform(dummy_preds)[:, target_idx]

        except Exception as e: st.warning(f"LSTM prediction failed: {e}")
    
    # Plotting
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=df.index, y=df[selected_energy], name='Actual', line={'color': '#00e6e6'}))
    if arima_preds is not None: fig_forecast.add_trace(go.Scatter(x=forecast_dates, y=arima_preds, name='ARIMA', line={'color': '#ff007f'}))
    if lstm_preds is not None: fig_forecast.add_trace(go.Scatter(x=forecast_dates, y=lstm_preds, name='LSTM', line={'color': '#00ff66'}))
    if prophet_preds is not None: fig_forecast.add_trace(go.Scatter(x=forecast_dates, y=prophet_preds, name='Prophet', line={'color': '#ffcc00'}))
    fig_forecast.update_layout(
        title=f'{selected_energy} Forecasts ({forecast_horizon} Months)',
        plot_bgcolor='#2c2c2c', paper_bgcolor='#2c2c2c', font_color='#f0f2f6', template='plotly_dark',
        title_font_color='white',
        legend_font_color='white',
        legend_title_font_color='white'
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    st.markdown('<h3 class="animate__fadeIn">Model Performance (MSE on Test Set)</h3>', unsafe_allow_html=True)
    st.dataframe(results, use_container_width=True, column_config={"MSE": {"format": ".4f"}})

with tab3:
    st.markdown('<h3 class="animate__fadeIn">Correlation Heatmap</h3>', unsafe_allow_html=True)
    if os.path.exists('correlations_heatmap.png'):
        st.image('correlations_heatmap.png', use_container_width=True)
    else:
        st.warning("Correlation heatmap not found. Please run the main script.")
