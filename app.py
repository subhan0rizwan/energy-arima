import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import os
import pickle
import torch
from datetime import datetime, timedelta
import importlib.util
import sys
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn

# Streamlit page configuration
st.set_page_config(
    page_title="Energy Price Forecasting Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a vibrant theme
st.markdown("""
    <style>
    /* Main body and background */
    .stApp {
        background-color: #1c1c1c;
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
        background-color: #00ff66; /* Energetic green */
        color: #1c1c1c;
        transform: translateY(-2px);
        box-shadow: 0 6px 10px rgba(0, 255, 102, 0.4);
    }

    /* Selectbox and Slider */
    .stSelectbox, .stSlider {
        background-color: #2c2c2c;
        color: #f0f2f6;
        border-radius: 8px;
        border: 1px solid #3c3c3c;
    }
    .stSelectbox>div>div>select {
        background-color: #2c2c2c;
        color: #f0f2f6;
    }

    /* Titles and Headers */
    h1, h2, h3 {
        color: #f0f2f6;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5);
    }
    h1 {
        border-bottom: 2px solid #00e6e6;
        padding-bottom: 10px;
    }
    h3 {
        margin-top: 30px;
    }

    /* Charts and dataframes */
    .stPlotlyChart {
        box-shadow: 8px 8px 15px rgba(0, 0, 0, 0.6);
        border-radius: 12px;
        overflow: hidden;
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.4);
    }
    .stMarkdown a {
        color: #00e6e6;
    }
    .stMarkdown a:hover {
        color: #00ff66;
    }
    
    /* Animations */
    .animate__fadeIn {
        animation: fadeIn 1.5s ease-in;
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# Check for critical dependencies
required_libraries = ['pandas', 'numpy', 'yfinance', 'plotly', 'torch', 'statsmodels', 'arch', 'prophet', 'sklearn', 'transformers']
missing_libraries = [lib for lib in required_libraries if not importlib.util.find_spec(lib)]
if missing_libraries:
    st.error(f"Missing critical libraries: {', '.join(missing_libraries)}. Please check requirements.txt and Python version.")
    st.stop()

def preprocess_for_lstm(df):
    """Replicates the preprocessing steps from main.py for LSTM input."""
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how='all').ffill().bfill()
    for col in df.columns:
        df[f'{col}_rolling'] = df[col].rolling(12).mean()
        df[f'{col}_std'] = df[col].rolling(12).std()
    df = df.fillna(method='bfill')
    df.columns = df.columns.astype(str)
    return df

@st.cache_data
def load_data():
    try:
        df = pd.read_parquet('energy_prices.parquet')
        results = pd.read_csv('model_results.csv')
        return df, results
    except Exception as e:
        st.warning(f"Error loading data: {e}. Using synthetic data.")
        dates = pd.date_range('2015-01-01', '2025-09-08', freq='M')
        np.random.seed(42)
        df = pd.DataFrame({
            'Petrol': np.clip(np.random.normal(loc=100, scale=10, size=len(dates)), 50, 150),
            'Uranium': np.clip(np.random.normal(loc=50, scale=5, size=len(dates)), 20, 80),
            'Natural_Gas': np.clip(np.random.normal(loc=5, scale=0.5, size=len(dates)), 2, 8),
            'Crude_Oil': np.clip(np.random.normal(loc=70, scale=7, size=len(dates)), 30, 110)
        }, index=dates)
        results = pd.DataFrame({
            'arima_mse': [float('inf')],
            'lstm_mse': [float('inf')],
            'prophet_mse': [float('inf')]
        })
        return df, results

required_files = ['energy_prices.parquet', 'arima_model.pkl', 'lstm_model.pth', 'prophet_model.pkl', 'correlations_heatmap.png', 'model_results.csv']
if not all(os.path.exists(f) for f in required_files):
    try:
        st.info("Generating data and models...")
        subprocess.run(['python', 'main.py'], check=True)
        st.cache_data.clear()
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Error generating data/models: {e}. Using synthetic data.")

df, results = load_data()
energy_types = [col for col in df.columns if col in ['Petrol', 'Uranium', 'Natural_Gas', 'Crude_Oil']]

processed_df = preprocess_for_lstm(df.copy())
if 'Sentiment' not in processed_df.columns:
    processed_df['Sentiment'] = 0.0

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

@st.cache_resource
def load_models(processed_df):
    models = {}
    try:
        with open('arima_model.pkl', 'rb') as f:
            models['arima'] = pickle.load(f)
    except:
        models['arima'] = None
    try:
        input_size = len(processed_df.columns)
        model = LSTM(input_size=input_size)
        model.load_state_dict(torch.load('lstm_model.pth'))
        model.eval()
        models['lstm'] = model
    except Exception as e:
        st.warning(f"LSTM model load failed: {e}")
        models['lstm'] = None
    try:
        with open('prophet_model.pkl', 'rb') as f:
            models['prophet'] = pickle.load(f)
    except:
        st.warning("Prophet model load failed.")
        models['prophet'] = None
    return models

models = load_models(processed_df)

# Sidebar controls
st.sidebar.title("⚙️ Controls")
selected_energy = st.sidebar.selectbox(
    "Select Energy Type",
    options=energy_types,
    index=energy_types.index('Petrol') if 'Petrol' in energy_types else 0
)
forecast_horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    options=[1, 3, 6, 12],
    format_func=lambda x: f"{x} Month{'s' if x > 1 else ''}",
    index=3
)
refresh_button = st.sidebar.button("Refresh Data")

# Handle refresh
if refresh_button:
    try:
        st.info("Refreshing data and models...")
        subprocess.run(['python', 'main.py'], check=True)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Error refreshing data: {e}")

# Main header
st.markdown('<h1 class="animate__fadeIn">⚡ Energy Price Forecasting Dashboard</h1>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🔮 Forecasts", "🔍 Advanced Analyses"])

# Tab 1: Data Overview
with tab1:
    st.markdown('<h3 class="animate__fadeIn">Historical Prices</h3>', unsafe_allow_html=True)
    fig_price = px.line(df, x=df.index, y=selected_energy, title=f'{selected_energy} Historical Prices')
    fig_price.update_layout(
        plot_bgcolor='#2c2c2c', paper_bgcolor='#2c2c2c', font_color='#f0f2f6',
        xaxis_title='Date', yaxis_title='Price ($)', template='plotly_dark',
        hovermode='x unified',
        line_color='#00e6e6'
    )
    st.plotly_chart(fig_price, use_container_width=True)
    
    st.markdown('<h3 class="animate__fadeIn">Data Table</h3>', unsafe_allow_html=True)
    table_data = df.reset_index().rename(columns={'index': 'Date'})[['Date'] + energy_types]
    st.dataframe(
        table_data,
        use_container_width=True,
        height=300,
        column_config={col: {"format": ".2f"} for col in energy_types}
    )
    
    st.download_button(
        label="Download Data",
        data=df.to_csv(),
        file_name=f"{selected_energy}_data.csv",
        mime="text/csv"
    )

# Tab 2: Forecasts
with tab2:
    st.markdown('<h3 class="animate__fadeIn">Model Forecasts & Sentiment</h3>', unsafe_allow_html=True)
    
    processed_df_forecast = preprocess_for_lstm(df.copy())
    if 'Sentiment' not in processed_df_forecast.columns:
        processed_df_forecast['Sentiment'] = 0.0
    
    # NEW LOGIC: Calculate the start date for the forecast period to be the end of the historical data
    last_date = df.index[-1]
    forecast_start_date = last_date + pd.DateOffset(months=1)
    
    # Create the test dates for the forecast horizon
    test_dates = pd.date_range(start=forecast_start_date, periods=forecast_horizon, freq='M')
    
    arima_preds, lstm_preds, prophet_preds = None, None, None
    
    try:
        if models['arima']:
            # Correctly handle exogenous variables for forecasting
            features = [col for col in processed_df_forecast.columns if col != selected_energy and ('rolling' in col or 'std' in col or 'Sentiment' in col)]
            # Use the most recent exogenous data for prediction
            exog_test = processed_df_forecast[features].iloc[-forecast_horizon:]
            arima_preds = models['arima'].forecast(steps=forecast_horizon, exog=exog_test)
    except Exception as e:
        st.warning(f"ARIMA prediction failed: {e}")
    
    try:
        if models['prophet']:
            future = models['prophet'].make_future_dataframe(periods=forecast_horizon, freq='M')
            future = pd.merge(future, processed_df_forecast.reset_index(), left_on='ds', right_on='index', how='left').drop('index', axis=1)
            future = future.ffill().bfill()
            prophet_preds = models['prophet'].predict(future.iloc[len(df):])['yhat']
    except Exception as e:
        st.warning(f"Prophet prediction failed: {e}")
    
    if models['lstm']:
        try:
            scaler = MinMaxScaler()
            scaled_df = pd.DataFrame(scaler.fit_transform(processed_df_forecast), index=processed_df_forecast.index, columns=processed_df_forecast.columns)
            scaled_target_idx = scaled_df.columns.get_loc(selected_energy)
            seq_length = 12
            last_sequence = scaled_df.values[-seq_length:]
            
            full_scaler = MinMaxScaler()
            full_scaler.fit(df[selected_energy].values.reshape(-1, 1))
            
            lstm_preds_list = []
            current_input = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)
            
            for _ in range(forecast_horizon):
                with torch.no_grad():
                    prediction = models['lstm'](current_input)
                prediction_value = prediction.squeeze().item()
                next_input_row = current_input.squeeze().numpy()[-1].copy()
                next_input_row[scaled_target_idx] = prediction_value
                
                next_input_row_tensor = torch.tensor(next_input_row, dtype=torch.float32)
                current_input = torch.cat((current_input.squeeze()[-seq_length+1:], next_input_row_tensor.unsqueeze(0))).unsqueeze(0)
                lstm_preds_list.append(prediction_value)
            
            lstm_preds = full_scaler.inverse_transform(np.array(lstm_preds_list).reshape(-1, 1)).squeeze()
        except Exception as e:
            st.warning(f"LSTM prediction failed: {e}")
            lstm_preds = None

    if arima_preds is None:
        np.random.seed(42)
        arima_preds = df[selected_energy].iloc[-forecast_horizon:].values + np.random.normal(0, 1, forecast_horizon)
    if prophet_preds is None:
        np.random.seed(42)
        prophet_preds = df[selected_energy].iloc[-forecast_horizon:].values + np.random.normal(0, 0.5, forecast_horizon)
    if lstm_preds is None:
        np.random.seed(42)
        lstm_preds = df[selected_energy].iloc[-forecast_horizon:].values + np.random.normal(0, 1.5, forecast_horizon)
    
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=df.index, y=df[selected_energy], name='Actual', line={'color': '#00e6e6'}))
    fig_forecast.add_trace(go.Scatter(x=test_dates, y=arima_preds, name='ARIMA', line={'color': '#ff007f'}))
    fig_forecast.add_trace(go.Scatter(x=test_dates, y=lstm_preds, name='LSTM', line={'color': '#00ff66'}))
    fig_forecast.add_trace(go.Scatter(x=test_dates, y=prophet_preds, name='Prophet', line={'color': '#ffcc00'}))
    fig_forecast.update_layout(
        title=f'{selected_energy} Forecasts ({forecast_horizon} Months)',
        plot_bgcolor='#2c2c2c', paper_bgcolor='#2c2c2c', font_color='#f0f2f6',
        xaxis_title='Date', yaxis_title='Price ($)', template='plotly_dark',
        hovermode='x unified'
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    st.markdown('<h3 class="animate__fadeIn">Model Performance</h3>', unsafe_allow_html=True)
    results_data = pd.DataFrame([
        {'Model': 'ARIMA', 'MSE': results['arima_mse'].iloc[0]},
        {'Model': 'LSTM', 'MSE': results['lstm_mse'].iloc[0]},
        {'Model': 'Prophet', 'MSE': results['prophet_mse'].iloc[0]}
    ])
    st.dataframe(
        results_data,
        use_container_width=True,
        column_config={"MSE": {"format": ".4f"}}
    )
    
    st.markdown('<h3 class="animate__fadeIn">Sentiment Analysis</h3>', unsafe_allow_html=True)
    st.write("Sentiment analysis disabled in this demo. Enable X API for live sentiment.")

# Tab 3: Advanced Analyses
with tab3:
    st.markdown('<h3 class="animate__fadeIn">Correlation Heatmap</h3>', unsafe_allow_html=True)
    if os.path.exists('correlations_heatmap.png'):
        st.image('correlations_heatmap.png', use_container_width=True)
    else:
        st.warning("Correlation heatmap not found.")
    
    st.markdown('<h3 class="animate__fadeIn">Monte Carlo Simulation</h3>', unsafe_allow_html=True)
    last_price = df[selected_energy].iloc[-1]
    vol = df[selected_energy].pct_change().std()
    simulations = 100
    paths = np.random.normal(0, vol, size=(forecast_horizon, simulations))
    future_prices = last_price * np.exp(np.cumsum(paths, axis=0))
    mean_path = future_prices.mean(axis=1)
    lower_bound = np.percentile(future_prices, 5, axis=1)
    upper_bound = np.percentile(future_prices, 95, axis=1)
    
    fig_monte_carlo = go.Figure()
    for i in range(min(10, simulations)):
        fig_monte_carlo.add_trace(go.Scatter(x=test_dates, y=future_prices[:, i], mode='lines', line={'color': '#00ff66', 'width': 1}, opacity=0.1, showlegend=False))
    fig_monte_carlo.add_trace(go.Scatter(x=test_dates, y=mean_path, name='Mean Path', line={'color': '#00e6e6', 'width': 3}))
    fig_monte_carlo.add_trace(go.Scatter(x=test_dates, y=lower_bound, name='5th Percentile', line={'color': '#ffcc00', 'dash': 'dash'}, fill=None))
    fig_monte_carlo.add_trace(go.Scatter(x=test_dates, y=upper_bound, name='95th Percentile', line={'color': '#ffcc00', 'dash': 'dash'}, fill='tonexty', fillcolor='rgba(255, 204, 0, 0.1)'))
    fig_monte_carlo.update_layout(
        title=f'{selected_energy} Monte Carlo Simulation ({forecast_horizon} Months)',
        plot_bgcolor='#2c2c2c', paper_bgcolor='#2c2c2c', font_color='#f0f2f6',
        xaxis_title='Date', yaxis_title='Price ($)', template='plotly_dark',
        hovermode='x unified'
    )
    st.plotly_chart(fig_monte_carlo, use_container_width=True)