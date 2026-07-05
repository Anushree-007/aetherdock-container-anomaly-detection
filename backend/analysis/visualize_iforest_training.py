# import pandas as pd
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# # === Load trained assets ===
# model = joblib.load("models/isolation_forest.pkl")
# scaler = joblib.load("models/scaler.pkl")
# feature_cols = joblib.load("models/feature_cols.pkl")

# # === Load dataset ===
# DATA_PATH = "container_metrics.csv"
# df = pd.read_csv(DATA_PATH)
# print(f"Loaded dataset: {len(df)} rows")

# # === Preprocess ===
# if np.issubdtype(df['timestamp'].dtype, np.number):
#     df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
# else:
#     df['timestamp'] = pd.to_datetime(df['timestamp'])

# df['time_window'] = df['timestamp'].dt.floor('5min')
# df['mem_used_mb'] = df['mem_used_bytes'] / (1024 * 1024)
# df['net_tx_mb'] = df['net_tx_bytes'] / (1024 * 1024)
# df['net_rx_mb'] = df['net_rx_bytes'] / (1024 * 1024)

# features = []
# for cname, group in df.groupby('container_name'):
#     agg = group.groupby('time_window').agg({
#         'cpu_percent': ['mean', 'max', 'std'],
#         'mem_used_mb': ['mean', 'max', 'std'],
#         'net_tx_mb': 'mean',
#         'net_rx_mb': 'mean'
#     })

#     # Flatten multi-level columns
#     agg.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in agg.columns]
#     agg.reset_index(inplace=True)

#     # ✅ Rename columns to match training features
#     agg.rename(columns={
#         'cpu_percent_mean': 'cpu_mean',
#         'cpu_percent_max': 'cpu_max',
#         'cpu_percent_std': 'cpu_std',
#         'mem_used_mb_mean': 'mem_mean_mb',
#         'mem_used_mb_max': 'mem_max_mb',
#         'mem_used_mb_std': 'mem_std_mb',
#         'net_tx_mb_mean': 'net_tx_mean_mb',
#         'net_rx_mb_mean': 'net_rx_mean_mb'
#     }, inplace=True)

#     # Compute slopes for trends
#     if len(agg) > 1:
#         agg['cpu_slope'] = np.gradient(agg['cpu_mean'])
#         agg['mem_slope'] = np.gradient(agg['mem_mean_mb'])
#     else:
#         agg['cpu_slope'] = 0
#         agg['mem_slope'] = 0

#     # Deviations from container mean
#     agg['cpu_dev'] = agg['cpu_mean'] - agg['cpu_mean'].mean()
#     agg['mem_dev'] = agg['mem_mean_mb'] - agg['mem_mean_mb'].mean()

#     # Simple z-scores (standardization per container)
#     agg['cpu_z'] = (agg['cpu_mean'] - agg['cpu_mean'].mean()) / (agg['cpu_mean'].std() + 1e-6)
#     agg['mem_z'] = (agg['mem_mean_mb'] - agg['mem_mean_mb'].mean()) / (agg['mem_mean_mb'].std() + 1e-6)

#     agg['container_name'] = cname
#     features.append(agg)

# features_df = pd.concat(features, ignore_index=True)
# print(f"✅ Feature dataframe ready: {features_df.shape}")

# # === Scale features ===
# X = features_df[feature_cols].fillna(0)
# X_scaled = scaler.transform(X)

# # === Predict anomalies ===
# scores = model.decision_function(X_scaled)
# preds = model.predict(X_scaled)
# features_df["anomaly_score"] = scores
# features_df["prediction"] = preds
# features_df["status"] = features_df["prediction"].map({1: "Normal", -1: "Anomaly"})

# # === PCA projection for visualization ===
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)

# features_df["pca1"] = X_pca[:, 0]
# features_df["pca2"] = X_pca[:, 1]

# # === Scatter plot with legend ===
# plt.figure(figsize=(10, 7))
# for cname, subset in features_df.groupby("container_name"):
#     plt.scatter(subset["pca1"], subset["pca2"],
#                 label=cname,
#                 alpha=0.6,
#                 s=20)

# plt.title("Container Behavior Visualization (PCA Projection)")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
# plt.tight_layout()
# plt.show()

# # === Histogram of anomaly scores ===
# plt.figure(figsize=(8, 5))
# plt.hist(scores, bins=50)
# plt.title("Distribution of Isolation Forest Anomaly Scores")
# plt.xlabel("Anomaly Score")
# plt.ylabel("Count")
# plt.show()

# # === Container-wise anomaly rate ===
# anomaly_summary = features_df.groupby("container_name")["status"].value_counts().unstack().fillna(0)
# print("\nContainer-wise anomaly summary:")
# print(anomaly_summary)





import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# === Load trained assets ===
model = joblib.load("models/isolation_forest.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

DATA_PATH = "container_metrics.csv"
BASELINE_PATH = "models/container_stats_baseline.csv"

# === Load dataset ===
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {len(df)} rows")

# === Preprocess ===
if np.issubdtype(df['timestamp'].dtype, np.number):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
else:
    df['timestamp'] = pd.to_datetime(df['timestamp'])

df['time_window'] = df['timestamp'].dt.floor('5min')
df['mem_used_mb'] = df['mem_used_bytes'] / (1024 * 1024)
df['net_tx_mb'] = df['net_tx_bytes'] / (1024 * 1024)
df['net_rx_mb'] = df['net_rx_bytes'] / (1024 * 1024)

# === Aggregate into 5-minute windows ===
features = []
for cname, group in df.groupby('container_name'):
    agg = group.groupby('time_window').agg({
        'cpu_percent': ['mean', 'max', 'std'],
        'mem_used_mb': ['mean', 'max', 'std'],
        'net_tx_mb': 'mean',
        'net_rx_mb': 'mean'
    })
    agg.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in agg.columns]
    agg.reset_index(inplace=True)

    # Rename to match model feature names
    agg.rename(columns={
        'cpu_percent_mean': 'cpu_mean',
        'cpu_percent_max': 'cpu_max',
        'cpu_percent_std': 'cpu_std',
        'mem_used_mb_mean': 'mem_mean_mb',
        'mem_used_mb_max': 'mem_max_mb',
        'mem_used_mb_std': 'mem_std_mb',
        'net_tx_mb_mean': 'net_tx_mean_mb',
        'net_rx_mb_mean': 'net_rx_mean_mb'
    }, inplace=True)

    if len(agg) > 1:
        agg['cpu_slope'] = np.gradient(agg['cpu_mean'])
        agg['mem_slope'] = np.gradient(agg['mem_mean_mb'])
    else:
        agg['cpu_slope'] = 0
        agg['mem_slope'] = 0

    agg['cpu_dev'] = agg['cpu_mean'] - agg['cpu_mean'].mean()
    agg['mem_dev'] = agg['mem_mean_mb'] - agg['mem_mean_mb'].mean()
    agg['cpu_z'] = (agg['cpu_mean'] - agg['cpu_mean'].mean()) / (agg['cpu_mean'].std() + 1e-6)
    agg['mem_z'] = (agg['mem_mean_mb'] - agg['mem_mean_mb'].mean()) / (agg['mem_mean_mb'].std() + 1e-6)
    agg['container_name'] = cname
    features.append(agg)

features_df = pd.concat(features, ignore_index=True)
print(f"✅ Feature dataframe ready: {features_df.shape}")

# === Scale features ===
X = features_df[feature_cols].fillna(0)
X_scaled = scaler.transform(X)

# === Predict anomalies ===
scores = model.decision_function(X_scaled)
preds = model.predict(X_scaled)
features_df["anomaly_score"] = scores
features_df["prediction"] = preds
features_df["status"] = features_df["prediction"].map({1: "Normal", -1: "Anomaly"})

# === PCA visualization ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
features_df["pca1"] = X_pca[:, 0]
features_df["pca2"] = X_pca[:, 1]

plt.figure(figsize=(10, 7))
for cname, subset in features_df.groupby("container_name"):
    plt.scatter(subset["pca1"], subset["pca2"], label=cname, alpha=0.6, s=20)
plt.title("Container Behavior (PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.show()

# === Histogram of anomaly scores ===
plt.figure(figsize=(8, 5))
plt.hist(scores, bins=50)
plt.title("Distribution of Isolation Forest Anomaly Scores")
plt.xlabel("Anomaly Score")
plt.ylabel("Count")
plt.show()

# === Container-wise anomaly summary ===
summary = features_df.groupby("container_name")["status"].value_counts().unstack().fillna(0)
print("\nContainer-wise anomaly summary:\n", summary)

# === EWMA Baseline Visualization ===
try:
    baseline_df = pd.read_csv(BASELINE_PATH)
    print(f"\nLoaded baseline stats from {BASELINE_PATH}")
except FileNotFoundError:
    print("⚠️ Baseline file not found. Skipping EWMA plot.")
    baseline_df = None

if baseline_df is not None:
    # Choose a few representative containers
    containers_to_plot = ['nginx_server', 'mysql-latest', 'cpu_burner', 'mem_leak']
    for cname in containers_to_plot:
        if cname not in df['container_name'].unique():
            continue
        sub = df[df['container_name'] == cname].copy()
        sub.sort_values('timestamp', inplace=True)

        # EWMA of CPU and Memory
        sub['cpu_ewma'] = sub['cpu_percent'].ewm(span=10, adjust=False).mean()
        sub['mem_ewma'] = (sub['mem_used_bytes'] / (1024 * 1024)).ewm(span=10, adjust=False).mean()

        plt.figure(figsize=(10, 4))
        plt.plot(sub['timestamp'], sub['cpu_percent'], label='CPU Actual', alpha=0.5)
        plt.plot(sub['timestamp'], sub['cpu_ewma'], label='CPU EWMA (Baseline)', linewidth=2)
        plt.title(f"CPU Trend with EWMA Baseline - {cname}")
        plt.xlabel("Time")
        plt.ylabel("CPU %")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        mem_mb = sub['mem_used_bytes'] / (1024 * 1024)
        plt.plot(sub['timestamp'], mem_mb, label='Memory Actual', alpha=0.5)
        plt.plot(sub['timestamp'], sub['mem_ewma'], label='Memory EWMA (Baseline)', linewidth=2)
        plt.title(f"Memory Trend with EWMA Baseline - {cname}")
        plt.xlabel("Time")
        plt.ylabel("Memory (MB)")
        plt.legend()
        plt.tight_layout()
        plt.show()




# import pandas as pd
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# # === Load trained assets ===
# model = joblib.load("models/isolation_forest.pkl")
# scaler = joblib.load("models/scaler.pkl")
# feature_cols = joblib.load("models/feature_cols.pkl")

# DATA_PATH = "container_metrics.csv"
# BASELINE_PATH = "models/container_stats_baseline.csv"

# # === Load dataset ===
# df = pd.read_csv(DATA_PATH)
# print(f"Loaded dataset: {len(df)} rows")

# if np.issubdtype(df['timestamp'].dtype, np.number):
#     df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
# else:
#     df['timestamp'] = pd.to_datetime(df['timestamp'])

# df['time_window'] = df['timestamp'].dt.floor('5min')
# df['mem_used_mb'] = df['mem_used_bytes'] / (1024 * 1024)
# df['net_tx_mb'] = df['net_tx_bytes'] / (1024 * 1024)
# df['net_rx_mb'] = df['net_rx_bytes'] / (1024 * 1024)

# # === Aggregate ===
# features = []
# for cname, group in df.groupby('container_name'):
#     agg = group.groupby('time_window').agg({
#         'cpu_percent': ['mean', 'max', 'std'],
#         'mem_used_mb': ['mean', 'max', 'std'],
#         'net_tx_mb': 'mean',
#         'net_rx_mb': 'mean'
#     })
#     agg.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in agg.columns]
#     agg.reset_index(inplace=True)
#     agg.rename(columns={
#         'cpu_percent_mean': 'cpu_mean',
#         'cpu_percent_max': 'cpu_max',
#         'cpu_percent_std': 'cpu_std',
#         'mem_used_mb_mean': 'mem_mean_mb',
#         'mem_used_mb_max': 'mem_max_mb',
#         'mem_used_mb_std': 'mem_std_mb',
#         'net_tx_mb_mean': 'net_tx_mean_mb',
#         'net_rx_mb_mean': 'net_rx_mean_mb'
#     }, inplace=True)

#     if len(agg) > 1:
#         agg['cpu_slope'] = np.gradient(agg['cpu_mean'])
#         agg['mem_slope'] = np.gradient(agg['mem_mean_mb'])
#     else:
#         agg['cpu_slope'] = 0
#         agg['mem_slope'] = 0

#     agg['cpu_dev'] = agg['cpu_mean'] - agg['cpu_mean'].mean()
#     agg['mem_dev'] = agg['mem_mean_mb'] - agg['mem_mean_mb'].mean()
#     agg['cpu_z'] = (agg['cpu_mean'] - agg['cpu_mean'].mean()) / (agg['cpu_mean'].std() + 1e-6)
#     agg['mem_z'] = (agg['mem_mean_mb'] - agg['mem_mean_mb'].mean()) / (agg['mem_mean_mb'].std() + 1e-6)
#     agg['container_name'] = cname
#     features.append(agg)

# features_df = pd.concat(features, ignore_index=True)
# print(f"✅ Feature dataframe ready: {features_df.shape}")

# # === Scale + Predict ===
# X = features_df[feature_cols].fillna(0)
# X_scaled = scaler.transform(X)
# scores = model.decision_function(X_scaled)
# preds = model.predict(X_scaled)
# features_df["anomaly_score"] = scores
# features_df["prediction"] = preds
# features_df["status"] = features_df["prediction"].map({1: "Normal", -1: "Anomaly"})

# # === PCA visualization ===
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)
# features_df["pca1"] = X_pca[:, 0]
# features_df["pca2"] = X_pca[:, 1]

# plt.figure(figsize=(10, 7))
# for cname, subset in features_df.groupby("container_name"):
#     plt.scatter(subset["pca1"], subset["pca2"], label=cname, alpha=0.6, s=20)
# plt.title("Container Behavior (PCA Projection)")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
# plt.tight_layout()
# plt.show()

# # === Histogram of anomaly scores ===
# plt.figure(figsize=(8, 5))
# plt.hist(scores, bins=50)
# plt.title("Distribution of Isolation Forest Anomaly Scores")
# plt.xlabel("Anomaly Score")
# plt.ylabel("Count")
# plt.show()

# # === Container-wise anomaly summary ===
# summary = features_df.groupby("container_name")["status"].value_counts().unstack().fillna(0)
# print("\nContainer-wise anomaly summary:\n", summary)

# # === EWMA Baseline Visualization with ±2σ bands ===
# try:
#     baseline_df = pd.read_csv(BASELINE_PATH)
#     print(f"\nLoaded baseline stats from {BASELINE_PATH}")
# except FileNotFoundError:
#     print("⚠️ Baseline file not found. Skipping EWMA plot.")
#     baseline_df = None

# if baseline_df is not None:
#     containers_to_plot = ['nginx_server', 'mysql-latest', 'cpu_burner', 'mem_leak']
#     for cname in containers_to_plot:
#         if cname not in df['container_name'].unique():
#             continue
#         sub = df[df['container_name'] == cname].copy()
#         sub.sort_values('timestamp', inplace=True)

#         # EWMA baselines
#         sub['cpu_ewma'] = sub['cpu_percent'].ewm(span=10, adjust=False).mean()
#         sub['mem_mb'] = sub['mem_used_bytes'] / (1024 * 1024)
#         sub['mem_ewma'] = sub['mem_mb'].ewm(span=10, adjust=False).mean()

#         # Rolling standard deviation
#         sub['cpu_std'] = sub['cpu_percent'].ewm(span=10, adjust=False).std()
#         sub['mem_std'] = sub['mem_mb'].ewm(span=10, adjust=False).std()

#         # === CPU plot ===
#         plt.figure(figsize=(10, 4))
#         plt.plot(sub['timestamp'], sub['cpu_percent'], label='CPU Actual', alpha=0.5)
#         plt.plot(sub['timestamp'], sub['cpu_ewma'], label='CPU EWMA Baseline', linewidth=2)
#         plt.fill_between(sub['timestamp'],
#                          sub['cpu_ewma'] - 2 * sub['cpu_std'],
#                          sub['cpu_ewma'] + 2 * sub['cpu_std'],
#                          color='gray', alpha=0.2, label='±2σ Band')
#         plt.title(f"CPU Usage vs EWMA Baseline ±2σ - {cname}")
#         plt.xlabel("Time")
#         plt.ylabel("CPU %")
#         plt.legend()
#         plt.tight_layout()
#         plt.show()

#         # === Memory plot ===
#         plt.figure(figsize=(10, 4))
#         plt.plot(sub['timestamp'], sub['mem_mb'], label='Memory Actual (MB)', alpha=0.5)
#         plt.plot(sub['timestamp'], sub['mem_ewma'], label='Memory EWMA Baseline', linewidth=2)
#         plt.fill_between(sub['timestamp'],
#                          sub['mem_ewma'] - 2 * sub['mem_std'],
#                          sub['mem_ewma'] + 2 * sub['mem_std'],
#                          color='gray', alpha=0.2, label='±2σ Band')
#         plt.title(f"Memory Usage vs EWMA Baseline ±2σ - {cname}")
#         plt.xlabel("Time")
#         plt.ylabel("Memory (MB)")
#         plt.legend()
#         plt.tight_layout()
#         plt.show()
