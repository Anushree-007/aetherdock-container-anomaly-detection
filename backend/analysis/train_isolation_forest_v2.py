# train_isolation_forest_v2.py
"""
Train pipeline:
 - 1-min aggregation
 - EWMA baseline per-container
 - compute features (mean,max,std, slopes, deviation_from_ewma, per-container z)
 - StandardScaler
 - IsolationForest(contamination=0.01)
 - Save artifacts: model, scaler, feature_cols, baseline table
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime

# ---------- Config ----------
CSV_PATH = os.path.abspath("container_metrics.csv")
MODELS_DIR = os.path.abspath(os.path.join("models"))
os.makedirs(MODELS_DIR, exist_ok=True)

# Which containers to treat as NORMAL (training set).
# Edit this list to match containers you know are healthy. Include dummy_* etc.
NORMAL_CONTAINERS = [
    "nginx_server", "mysql-latest", "mysql-older", "redis_cache",
    "memcached_cache", "postgres_db", "rabbitmq_broker",
    "grafana_dash", "apache_server", "xenodochial_blackburn", 
    "elasticsearch_node", "mongo-express", "mongo", "wonderful_joliot", "agitated_sammet", "recursing_blackburn"
    "memcached_cache", "influxdb_tsdb"
]
# contamination for IsolationForest (start small)
CONTAMINATION = 0.15

# EWMA alpha
EWMA_ALPHA = 0.15

# aggregation window
TIME_WINDOW = "1min"

# minimum windows per container to include in training
MIN_WINDOWS = 10

# ----------------------------

print("Loading dataset:", CSV_PATH)
df = pd.read_csv(CSV_PATH)
print("Rows:", len(df), "Columns:", len(df.columns))

# convert timestamp (handles unix ints)
if np.issubdtype(df['timestamp'].dtype, np.number):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
else:
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# convert bytes -> MB
df['mem_used_mb'] = df['mem_used_bytes'] / (1024 * 1024)
df['net_tx_mb'] = df['net_tx_bytes'] / (1024 * 1024)
df['net_rx_mb'] = df['net_rx_bytes'] / (1024 * 1024)

# floor to 1-minute windows
df['time_window'] = df['timestamp'].dt.floor(TIME_WINDOW)

# Filter training set to known NORMAL containers
train_df = df[df['container_name'].isin(NORMAL_CONTAINERS)].copy()
print("Training rows after normal filter:", len(train_df))
if train_df.empty:
    print("WARNING: training dataframe empty after filtering NORMAL_CONTAINERS.")
    print("Adjust NORMAL_CONTAINERS list or provide more data.")
    # continue, but will probably fail later

# Aggregate per container_name + time_window
agg_list = []
for cname, group in train_df.groupby('container_name'):
    g = group.groupby('time_window').agg(
        cpu_mean = ('cpu_percent', 'mean'),
        cpu_max = ('cpu_percent', 'max'),
        cpu_std = ('cpu_percent', 'std'),
        mem_mean_mb = ('mem_used_mb', 'mean'),
        mem_max_mb = ('mem_used_mb', 'max'),
        mem_std_mb = ('mem_used_mb', 'std'),
        net_tx_mean_mb = ('net_tx_mb', 'mean'),
        net_rx_mean_mb = ('net_rx_mb', 'mean'),
        count = ('cpu_percent', 'count')
    ).reset_index().sort_values('time_window')

    # if too few windows skip container for stable baseline
    if len(g) < MIN_WINDOWS:
        print(f"Skipping {cname}: only {len(g)} windows (<{MIN_WINDOWS})")
        continue

    # slopes: gradient on cpu_mean and mem_mean_mb
    # np.gradient returns slope per point (proxy)
    g['cpu_slope'] = np.gradient(g['cpu_mean'])  # units: cpu percent per window step
    g['mem_slope'] = np.gradient(g['mem_mean_mb'])  # units: MB per window step

    # EWMA baseline per container on cpu_mean and mem_mean_mb
    g['cpu_ewma'] = g['cpu_mean'].ewm(alpha=EWMA_ALPHA, adjust=False).mean()
    g['mem_ewma'] = g['mem_mean_mb'].ewm(alpha=EWMA_ALPHA, adjust=False).mean()

    # deviation from ewma
    g['cpu_dev'] = g['cpu_mean'] - g['cpu_ewma']
    g['mem_dev'] = g['mem_mean_mb'] - g['mem_ewma']

    # add container_name
    g['container_name'] = cname

    agg_list.append(g)

if len(agg_list) == 0:
    raise RuntimeError("No containers had enough data for training. Increase data or lower MIN_WINDOWS.")

features_df = pd.concat(agg_list, ignore_index=True)
print("Feature rows:", len(features_df))

# Per-container normalization: compute mean/std for cpu_mean and mem_mean_mb (within each container)
container_stats = features_df.groupby('container_name').agg(
    cpu_mean_container = ('cpu_mean', 'mean'),
    cpu_std_container = ('cpu_mean', 'std'),
    mem_mean_container = ('mem_mean_mb', 'mean'),
    mem_std_container = ('mem_mean_mb', 'std')
).reset_index()
# fill zeros to avoid div by zero
container_stats['cpu_std_container'].replace(0, 1e-6, inplace=True)
container_stats['mem_std_container'].replace(0, 1e-6, inplace=True)

# join stats back
features_df = features_df.merge(container_stats, on='container_name', how='left')

# z-scores per container
features_df['cpu_z'] = (features_df['cpu_mean'] - features_df['cpu_mean_container']) / features_df['cpu_std_container']
features_df['mem_z'] = (features_df['mem_mean_mb'] - features_df['mem_mean_container']) / features_df['mem_std_container']

# select features for model
feature_cols = [
    'cpu_mean', 'cpu_max', 'cpu_std',
    'mem_mean_mb', 'mem_max_mb', 'mem_std_mb',
    'net_tx_mean_mb', 'net_rx_mean_mb',
    'cpu_slope', 'mem_slope',
    'cpu_dev', 'mem_dev',
    'cpu_z', 'mem_z'
]

# fill NaNs
X = features_df[feature_cols].fillna(0).astype(float)

print("Feature matrix shape:", X.shape)
print("Feature columns:", feature_cols)

# Global scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest
model = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    random_state=42,
    n_jobs=-1
)
model.fit(X_scaled)

# Scores & predictions for diagnostics
scores = model.decision_function(X_scaled)  # higher = more normal
preds = model.predict(X_scaled)  # 1 normal, -1 anomaly
features_df['anomaly_score'] = scores
features_df['prediction'] = preds
features_df['status'] = features_df['prediction'].map({1: "Normal", -1: "Anomaly"})

# save artifacts
model_path = os.path.join(MODELS_DIR, "isolation_forest.pkl")
scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
features_path = os.path.join(MODELS_DIR, "feature_cols.pkl")
baseline_path = os.path.join(MODELS_DIR, "container_stats_baseline.csv")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(feature_cols, features_path)
container_stats.to_csv(baseline_path, index=False)

print("Saved model to:", model_path)
print("Saved scaler to:", scaler_path)
print("Saved feature cols to:", features_path)
print("Saved baseline table to:", baseline_path)

# Diagnostics: anomaly counts per container
summary = features_df.groupby('container_name')['status'].value_counts().unstack(fill_value=0)
print("\nAnomaly summary (train containers):\n", summary.head(20))

# If you have a set of KNOWN anomaly containers (holdout), evaluate them:
KNOWN_ANOMALIES = ["cpu_load", "cpu_burner", "mem_leak", "crash_loop", "net_flood", "disk_stress"]
print("\nEvaluating known anomaly containers (if present in original df):")
holdout_results = []
for cname in KNOWN_ANOMALIES:
    subset = df[df['container_name'] == cname]
    if subset.empty:
        continue
    # create same aggregated features for the holdout container
    s = subset.copy()
    s['time_window'] = s['timestamp'].dt.floor(TIME_WINDOW)
    agg = s.groupby('time_window').agg(
        cpu_mean = ('cpu_percent', 'mean'),
        cpu_max = ('cpu_percent', 'max'),
        cpu_std = ('cpu_percent', 'std'),
        mem_mean_mb = ('mem_used_mb', 'mean'),
        mem_max_mb = ('mem_used_mb', 'max'),
        mem_std_mb = ('mem_used_mb', 'std'),
        net_tx_mean_mb = ('net_tx_mb', 'mean'),
        net_rx_mean_mb = ('net_rx_mb', 'mean'),
    ).reset_index().sort_values('time_window')
    if len(agg) < 2:
        continue
    agg['cpu_slope'] = np.gradient(agg['cpu_mean'])
    agg['mem_slope'] = np.gradient(agg['mem_mean_mb'])
    # compute EWMA using same alpha
    agg['cpu_ewma'] = agg['cpu_mean'].ewm(alpha=EWMA_ALPHA, adjust=False).mean()
    agg['mem_ewma'] = agg['mem_mean_mb'].ewm(alpha=EWMA_ALPHA, adjust=False).mean()
    agg['cpu_dev'] = agg['cpu_mean'] - agg['cpu_ewma']
    agg['mem_dev'] = agg['mem_mean_mb'] - agg['mem_ewma']
    # per-container z using training container_stats mean/std if container_stats has a row for that name; else use agg mean/std
    if cname in container_stats['container_name'].values:
        row = container_stats[container_stats['container_name'] == cname].iloc[0]
        cpu_mean_c = row['cpu_mean_container']
        cpu_std_c = row['cpu_std_container'] if row['cpu_std_container'] > 0 else 1e-6
        mem_mean_c = row['mem_mean_container']
        mem_std_c = row['mem_std_container'] if row['mem_std_container'] > 0 else 1e-6
    else:
        cpu_mean_c = agg['cpu_mean'].mean()
        cpu_std_c = agg['cpu_mean'].std() if agg['cpu_mean'].std() > 0 else 1e-6
        mem_mean_c = agg['mem_mean_mb'].mean()
        mem_std_c = agg['mem_mean_mb'].std() if agg['mem_mean_mb'].std() > 0 else 1e-6
    agg['cpu_z'] = (agg['cpu_mean'] - cpu_mean_c) / cpu_std_c
    agg['mem_z'] = (agg['mem_mean_mb'] - mem_mean_c) / mem_std_c

    # build X_holdout following feature_cols
    agg_X = agg.reindex(columns=feature_cols).fillna(0).astype(float)
    agg_X_scaled = scaler.transform(agg_X)
    holdout_scores = model.decision_function(agg_X_scaled)
    holdout_preds = model.predict(agg_X_scaled)
    anomaly_frac = (holdout_preds == -1).mean()
    print(f"Holdout {cname}: windows={len(agg_X)}, anomaly_fraction={anomaly_frac:.3f}, mean_score={holdout_scores.mean():.3f}")
    holdout_results.append((cname, len(agg_X), anomaly_frac, holdout_scores.mean()))

print("\nTraining complete.")
