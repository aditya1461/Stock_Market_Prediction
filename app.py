import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import os
import datetime
import plotly.graph_objs as go

# Streamlit configuration (remove theme parameter)
st.set_page_config(page_title="Stock Market Predictor", page_icon="📈", layout="wide", initial_sidebar_state="auto")

# Load the trained model (Ensure correct path)
model_path = r"C:\Users\adity\OneDrive\Documents\Desktop\Jupyter Projects\adityaa\adityaa\models\StockPricePrediction.h5"

# Check if model file exists
if not os.path.exists(model_path):
    st.error(f"⚠️ Model file not found at: {model_path}")
    st.stop()

model = load_model(model_path)

# Streamlit UI
st.title('📈 Stock Market Predictor')

# Light/Dark mode toggle (using CSS workaround)
mode = st.sidebar.selectbox("Select Mode", ["Light", "Dark"])

# Apply custom CSS for theming
if mode == "Light":
    st.markdown(
        """
        <style>
        body {
            background-color: white;
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        body {
            background-color: #1E1E1E;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Custom Date Range Selector
start_date = st.date_input('Start Date', datetime.date(2012, 1, 1))
end_date = st.date_input('End Date', datetime.date.today())

# Input stock symbol
stock = st.text_input('Enter Stock Symbol (e.g., AAPL, TSLA, GOOG)', 'GOOG').upper()

# Fetch stock data
try:
    data = yf.download(stock, start=start_date, end=end_date)
    if data.empty:
        st.error(f"⚠️ No data found for {stock}. Please check the symbol.")
        st.stop()
except Exception as e:
    st.error(f"⚠️ Error fetching stock data: {e}")
    st.stop()

st.subheader(f'📊 Stock Data for {stock}')
st.write(data)  # Display stock data

# Define the lookback period (number of past days used for prediction)
lookback_period = 30  # Reduced from 100 to 30 for earlier prediction start
# Note: If you change the lookback period, you must retrain your model with the new lookback period,
# as the model's input shape (e.g., for LSTM) depends on this value.

# Scaling data for the entire dataset
scaler = MinMaxScaler(feature_range=(0,1))
data_full = data['Close']
data_full_scaled = scaler.fit_transform(data_full.values.reshape(-1,1))

# Prepare input features for prediction over the entire range
x_full = []
for i in range(lookback_period, len(data_full_scaled)):
    x_full.append(data_full_scaled[i-lookback_period:i])

x_full = np.array(x_full)

# Make predictions for the entire range
predicted_prices = model.predict(x_full)
predicted_prices = scaler.inverse_transform(predicted_prices).flatten()

# Match prediction dates (starting from the lookback_period-th day)
prediction_dates = data.index[lookback_period:]

# Calculate prediction start and end dates
prediction_start_date = prediction_dates[0].strftime('%Y-%m-%d')
prediction_end_date = prediction_dates[-1].strftime('%Y-%m-%d')
prediction_days = len(prediction_dates)

# Summary Stats
total_days = (data.index[-1] - data.index[0]).days + 1
highest_price = float(data['Close'].max())
lowest_price = float(data['Close'].min())

# Display summary
st.subheader("📌 Summary Stats for Selected Period")
st.write(f"1) Total number of days from start to end: **{total_days} days**")
st.write(f"2) Highest closing price in range: **${highest_price:.2f}**")
st.write(f"3) Lowest closing price in range: **${lowest_price:.2f}**")

# Display prediction period
st.subheader("📅 Prediction Period")
st.write(f"Prediction Start Date: **{prediction_start_date}**")
st.write(f"Prediction End Date: **{prediction_end_date}**")
st.write(f"Total Prediction Days: **{prediction_days} days**")

# Plotly Graph for Predictions
st.subheader(f'📉 Predicted Stock Prices for {stock}')

# Create a DataFrame for plotting
pred_df = pd.DataFrame({
    'Date': prediction_dates,
    'Predicted Price': predicted_prices
})

# Plotly line chart
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=pred_df['Date'],
    y=pred_df['Predicted Price'],
    mode='lines',
    line=dict(color='royalblue', width=2),
    name='Predicted Price'
))

fig.update_layout(
    title=f'📉 Predicted Stock Prices for {stock}',
    title_font=dict(size=20),
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white' if mode == "Dark" else 'black'),
    hovermode='x unified',
)

st.plotly_chart(fig, use_container_width=True)

# Download Button for CSV
st.subheader("📥 Download Stock Data as CSV")
csv = data.to_csv(index=True)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name=f"{stock}_stock_data.csv",
    mime="text/csv"
)

# Future Prediction (next 30 days)
st.subheader("🔮 Future Stock Price Prediction for the Next 30 Days")
future_days = 30

# Generate future predictions
last_lookback_days = data_full.tail(lookback_period).values.reshape(-1,1)
future_data = np.copy(last_lookback_days)

future_predictions = []

for _ in range(future_days):
    future_scaled = scaler.transform(future_data)
    future_input = future_scaled.reshape(1, lookback_period, 1)
    predicted_price = model.predict(future_input)
    future_predictions.append(predicted_price[0][0])
    
    future_data = np.append(future_data, predicted_price[0][0]).reshape(-1, 1)[1:]

# Convert to original scale
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

# Create future dates for prediction
last_date = data.index[-1]
future_dates = pd.date_range(last_date, periods=future_days + 1, freq='B')[1:]

# Plot future predictions
future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Price': future_predictions
})

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=future_df['Date'],
    y=future_df['Predicted Price'],
    mode='lines',
    line=dict(color='green', width=2),
    name='Future Prediction'
))

fig2.update_layout(
    title=f'🔮 Future Stock Prices for {stock} (Next 30 Days)',
    title_font=dict(size=20),
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white' if mode == "Dark" else 'black'),
    hovermode='x unified',
)

st.plotly_chart(fig2, use_container_width=True)

# Refresh button
if st.button("🔄 Refresh Data"):
    st.experimental_rerun()