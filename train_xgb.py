import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==== Load processed data ====
df = pd.read_csv("data/TSLA_processed.csv", index_col=0)
# Convert Volume to numeric, force errors to NaN
df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

df = df.dropna(subset=["Volume", "Target"])
# ==== Define features and label ====
features = ["Return", "LogReturn", "MA10", "MA50", "RSI", "MACD", "Volume"]
X = df[features]
y = df["Target"].astype(int)


# ==== Train-test split ====
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# ==== Train XGBoost model ====
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)

# ==== Predict and evaluate ====
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {accuracy:.2f}")
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n🧾 Classification Report:")
print(classification_report(y_test, y_pred))

# ==== Plot Confusion Matrix ====
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("XGBoost Confusion Matrix")
plt.tight_layout()
plt.show()

# ==== Save model ====
os.makedirs("models", exist_ok=True)
model.save_model("models/xgb_model.json")
print("\n💾 Model saved to models/xgb_model.json")
