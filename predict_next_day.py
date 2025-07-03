import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler

# ==== Load processed data ====
df = pd.read_csv("data/AAPL_processed.csv", index_col=0)

# ==== Define feature columns ====
features = ["Return", "LogReturn", "MA10", "MA50", "RSI", "MACD", "Volume"]

# ==== Drop any rows with missing values ====
df = df.dropna(subset=features)

# ==== Scale features (must match training scaling!) ====
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# ==== Get the latest (most recent) row ====
latest_data = df[features].iloc[-1:]
print("🔍 Latest input features:\n", latest_data)

# ==== Load trained model ====
model = xgb.XGBClassifier()
model.load_model("models/xgb_model.json")

# ==== Predict next day's direction ====
prediction = model.predict(latest_data)[0]
probability = model.predict_proba(latest_data)[0][1]

# ==== Show result ====
direction = "UP 📈" if prediction == 1 else "DOWN 📉"
print(f"\n✅ Predicted direction: {direction}")
print(f"🧠 Model confidence (probability of UP): {probability:.2f}")
