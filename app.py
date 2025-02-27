import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import datetime

# Load trained model
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Fetch stock data from Stooq
def fetch_stock_data_stooq(ticker, days=50):
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d"
    try:
        df = pd.read_csv(url)
        if df.empty:
            return None
        return df.tail(days).reset_index(drop=True)
    except Exception:
        return None

# Prepare dataset
def prepare_data(df):
    feature_cols = ["Open", "High", "Low", "Close", "Volume"]
    data = df[feature_cols].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

# Predict next day's closing price
def predict_next_day(model, data_scaled, seq_length, scaler):
    model.eval()
    last_seq = torch.tensor(data_scaled[-seq_length:], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        predicted_price_scaled = model(last_seq).item()
    predicted_price_actual = scaler.inverse_transform([[0, 0, 0, predicted_price_scaled, 0]])[0, 3]
    return predicted_price_actual

# Streamlit UI
st.title("US Stock Price Prediction")
st.sidebar.header("Enter Stock Code")

# User enters stock ticker
stock_ticker = st.sidebar.text_input("Stock Code (e.g., AAPL.US, TSLA.US):").strip().upper()

if stock_ticker:
    # Load trained model
    model_path = "stock_lstm_model.pth"
    input_size, hidden_size, num_layers = 5, 64, 2
    model = StockLSTM(input_size, hidden_size, num_layers)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.to(torch.device("cpu"))

    # Fetch data from Stooq
    df = fetch_stock_data_stooq(stock_ticker)
    
    if df is not None:
        st.write(f"### Last 5 Days Data for {stock_ticker}")
        st.dataframe(df.tail(5))
        
        seq_length = 30
        data_scaled, scaler = prepare_data(df)

        if len(data_scaled) >= seq_length:
            predicted_price = predict_next_day(model, data_scaled, seq_length, scaler)
            next_day = (pd.to_datetime(df["Date"].iloc[-1]) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            st.success(f"📅 **Predicted Closing Price for {stock_ticker} on {next_day}:** ${predicted_price:.2f}")
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            st.line_chart(df["Close"])
        else:
            st.warning("Not enough data for prediction. Try a different stock.")
    else:
        st.error(f"❌ No data available for {stock_ticker}. Try another stock.")