# backend/analysis/test_isolation_forest.py
import pandas as pd
import joblib
import numpy as np

# ==== Load model and feature columns ====
model = joblib.load("models/isolation_forest.pkl")
feature_cols = joblib.load("models/isolation_forest_features.pkl")

# ==== Load dataset again ====
df = pd.read_csv("container_metrics.csv")

# Convert timestamp if needed
if np.issubdtype(df['timestamp'].dtype, np.number):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
else:
    df['timestamp'] = pd.to_datetime(df['timestamp'])

df['time_window'] = df['timestamp'].dt.floor('5min')
df['mem_used_mb'] = df['mem_used_bytes'] / (1024 * 1024)
df['net_tx_mb'] = df['net_tx_bytes'] / (1024 * 1024)
df['net_rx_mb'] = df['net_rx_bytes'] / (1024 * 1024)

features = []
for cname, group in df.groupby('container_name'):
    agg = group.groupby('time_window').agg({
        'cpu_percent': 'mean',
        'mem_used_mb': 'mean',
        'net_tx_mb': 'mean',
        'net_rx_mb': 'mean'
    }).reset_index()

    if len(agg) > 1:
        agg['cpu_slope'] = np.gradient(agg['cpu_percent'])
        agg['mem_slope'] = np.gradient(agg['mem_used_mb'])
    else:
        agg['cpu_slope'] = 0
        agg['mem_slope'] = 0

    agg['container_name'] = cname
    features.append(agg)

features_df = pd.concat(features, ignore_index=True)

X = features_df[feature_cols].fillna(0)

# ==== Predict anomalies ====
preds = model.predict(X)
scores = model.decision_function(X)

# Add results back to dataframe
features_df['anomaly'] = preds
features_df['anomaly_score'] = scores

# Map -1 → Anomaly, 1 → Normal
features_df['status'] = features_df['anomaly'].map({-1: "Anomaly", 1: "Normal"})

# ==== Print summary ====
print("✅ Anomaly Scoring Complete!")
print(features_df[['container_name', 'time_window', 'status', 'anomaly_score']].head(20))

# ==== Optional: Save to CSV ====
features_df.to_csv("models/isolation_forest_results.csv", index=False)
print("📁 Results saved to models/isolation_forest_results.csv")
