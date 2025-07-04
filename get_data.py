import yfinance as yf
import pandas as pd
import numpy as np
import os
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import StandardScaler
import pickle

DATA_DIR = "data"

def download_data(ticker, start="2015-01-01", end="2024-12-31"):
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    return df

def add_features(df):
    df["Return"] = df["Close"].pct_change()
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))

    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()

    # Fix: Ensure "Close" is a 1D Series
    close_1d = df["Close"].squeeze()

    rsi = RSIIndicator(close_1d, window=14)
    df["RSI"] = rsi.rsi()

    macd = MACD(close_1d)
    df["MACD"] = macd.macd_diff()

    df.dropna(inplace=True)
    return df



def add_target(df):
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    return df
"""
def scale_features(df, feature_cols):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    df_scaled = pd.DataFrame(scaled, columns=feature_cols, index=df.index)
    df.update(df_scaled)
    return df, scaler
"""
#You shouldnt be scaling the whole dataset dumbass the mean and stand
def save_processed_data(df, ticker):
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(f"{DATA_DIR}/{ticker}_processed.csv")
    print(f"Saved processed data to {DATA_DIR}/{ticker}_processed.csv")

def run_pipeline(ticker):
    print(f"🔍 Fetching data for {ticker}")
    df = download_data(ticker)
    df = add_features(df)
    df = add_target(df)

    features = ["Return", "LogReturn", "MA10", "MA50", "RSI", "MACD", "Volume"]
    #df, scaler = scale_features(df, features)

    save_processed_data(df, ticker)

    # Save scaler too for later use in models
   
    #with open(f"{DATA_DIR}/{ticker}_scaler.pkl", "wb") as f:
   #     pickle.dump(scaler, f)

    return df

if __name__ == "__main__":
    df = run_pipeline("TSLA")
    print(df.tail())
