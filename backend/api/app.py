# backend/api/app.py
import os
import time
import threading
from collections import deque, defaultdict
from datetime import datetime
import math

from flask import Flask, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

# -----------------------
# Configuration
# -----------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))  # project root
MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_forest.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")  # optional; v2 training saves scaler
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_cols.pkl")
BASELINE_PATH = os.path.join(BASE_DIR, "models", "container_stats_baseline.csv")
RESULTS_CSV = os.path.join(BASE_DIR, "models", "isolation_forest_results.csv")

# live buffer settings
SAMPLE_INTERVAL = 10        # seconds between Docker stats samples (approx)
WINDOW_SECONDS = 300        # aggregation window length in seconds (e.g., last 5 minutes)
WINDOW_SIZE = max(6, int(WINDOW_SECONDS / SAMPLE_INTERVAL))  # number of samples to keep per container

POLL_INTERVAL = max(2, SAMPLE_INTERVAL)  # how often background thread polls docker
EWMA_SPAN = 5  # span used for EWMA smoothing in rolling computations

# -----------------------
# Flask app
# -----------------------
# app = Flask(__name__)
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend", "templates")
app = Flask(__name__, template_folder=FRONTEND_DIR)


# -----------------------
# Docker client (best-effort)
# -----------------------
try:
    import docker
    docker_client = docker.from_env()
    print("✅ Docker client initialized.")
except Exception as e:
    docker_client = None
    print("⚠️ Docker client not available:", e)

# -----------------------
# Load model, scaler, features, baseline
# -----------------------
model = None
scaler = None
feature_cols = None
baseline_df = None

def safe_load_artifacts():
    global model, scaler, feature_cols, baseline_df
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print("✅ Model loaded:", MODEL_PATH)
        else:
            print("⚠️ Model file not found at", MODEL_PATH)
            model = None
    except Exception as e:
        print("⚠️ Error loading model:", e)
        model = None

    try:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print("✅ Scaler loaded:", SCALER_PATH)
        else:
            scaler = None
            print("ℹ️ Scaler not found (OK if training didn't create one).")
    except Exception as e:
        print("⚠️ Error loading scaler:", e)
        scaler = None

    try:
        if os.path.exists(FEATURES_PATH):
            feature_cols = joblib.load(FEATURES_PATH)
            print("✅ Feature columns loaded:", FEATURES_PATH)
        else:
            feature_cols = None
            print("⚠️ Feature columns not found:", FEATURES_PATH)
    except Exception as e:
        print("⚠️ Error loading feature columns:", e)
        feature_cols = None

    try:
        if os.path.exists(BASELINE_PATH):
            baseline_df = pd.read_csv(BASELINE_PATH)
            # ensure container_name column exists
            if 'container_name' not in baseline_df.columns:
                baseline_df = None
                print("⚠️ Baseline CSV missing 'container_name' column.")
            else:
                baseline_df = baseline_df.set_index('container_name')
                print("✅ Baseline loaded:", BASELINE_PATH)
        else:
            baseline_df = None
            print("ℹ️ Baseline CSV not found (per-container normalization won't be available).")
    except Exception as e:
        baseline_df = None
        print("⚠️ Error loading baseline CSV:", e)

safe_load_artifacts()

# -----------------------
# In-memory rolling buffers:
# { container_id: deque([ {timestamp, cpu_percent, mem_used_mb, net_tx_mb, net_rx_mb, raw_stats...}, ... ]) }
# -----------------------
buffers = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
buffers_lock = threading.Lock()

# -----------------------
# Helper: safe docker stats extraction (robust to missing fields)
# -----------------------
def extract_stats_from_container(container):
    """
    Return a dict with normalized fields:
    {
      timestamp, container_id, container_name, cpu_percent, mem_used_mb, net_tx_mb, net_rx_mb, status, image
    }
    Returns None on failure.
    """
    try:
        stats = container.stats(stream=False)
    except Exception as e:
        # Sometimes Docker SDK fails (e.g., container removed). Don't crash.
        print(f"⚠️ Error calling stats() for {getattr(container,'name', '?')}: {e}")
        return None

    try:
        cpu_stats = stats.get("cpu_stats", {}) or {}
        precpu_stats = stats.get("precpu_stats", {}) or {}
        mem_stats = stats.get("memory_stats", {}) or {}

        # cpu usage deltas (robust)
        cpu_total = cpu_stats.get("cpu_usage", {}).get("total_usage", 0)
        precpu_total = precpu_stats.get("cpu_usage", {}).get("total_usage", 0)
        cpu_delta = cpu_total - precpu_total

        system_cpu = cpu_stats.get("system_cpu_usage", 0)
        pre_system_cpu = precpu_stats.get("system_cpu_usage", 0)
        system_delta = system_cpu - pre_system_cpu

        # fallback for per-CPU count
        percpu = cpu_stats.get("cpu_usage", {}).get("percpu_usage", [])
        cpu_count = len(percpu) if percpu else 1

        cpu_percent = 0.0
        if system_delta > 0 and cpu_delta > 0:
            cpu_percent = (cpu_delta / system_delta) * cpu_count * 100.0
            # clip to reasonable range
            if not math.isfinite(cpu_percent):
                cpu_percent = 0.0

        # memory
        mem_used = mem_stats.get("usage", 0)
        mem_limit = mem_stats.get("limit", 1)
        mem_used_mb = mem_used / (1024.0 * 1024.0)
        mem_limit_mb = mem_limit / (1024.0 * 1024.0)
        mem_percent = (mem_used / mem_limit) * 100.0 if mem_limit > 0 else 0.0

        # network: sum across interfaces (may be empty)
        networks = stats.get("networks", {}) or {}
        net_tx = 0
        net_rx = 0
        for iface, obj in networks.items():
            net_tx += obj.get("tx_bytes", 0)
            net_rx += obj.get("rx_bytes", 0)
        net_tx_mb = net_tx / (1024.0 * 1024.0)
        net_rx_mb = net_rx / (1024.0 * 1024.0)

        timestamp = datetime.utcnow().timestamp()

        image = "unknown"
        try:
            image = container.image.tags[0] if container.image and container.image.tags else "unknown"
        except Exception:
            pass

        status = getattr(container, "status", "unknown")

        return {
            "timestamp": timestamp,
            "container_id": container.id[:12],
            "container_name": container.name,
            "cpu_percent": float(cpu_percent),
            "mem_used_mb": float(mem_used_mb),
            "mem_limit_mb": float(mem_limit_mb),
            "mem_percent": float(mem_percent),
            "net_tx_mb": float(net_tx_mb),
            "net_rx_mb": float(net_rx_mb),
            "status": status,
            "image": image,
            "raw_stats": stats
        }

    except Exception as e:
        print("⚠️ Error extracting stats for", getattr(container, "name", "?"), e)
        return None

# -----------------------
# Background poller that refreshes rolling buffers for all running containers
# -----------------------
def background_poller():
    while True:
        try:
            if docker_client:
                containers = docker_client.containers.list(all=False)  # only running
                with buffers_lock:
                    # ensure buffers exist for running containers and remove buffers for deleted ones
                    running_ids = set()
                    for c in containers:
                        running_ids.add(c.id[:12])
                        stat = extract_stats_from_container(c)
                        if stat:
                            buffers[c.id[:12]].append(stat)
                    # remove old buffers for containers that are not running to free memory
                    for cid in list(buffers.keys()):
                        if cid not in running_ids:
                            # keep buffer but do not remove — keep for historical quick queries
                            # If you want to purge: del buffers[cid]
                            pass
            else:
                # no docker client available — do nothing
                pass
        except Exception as e:
            print("⚠️ background_poller error:", e)
        time.sleep(POLL_INTERVAL)

# start poller thread on import
poller_thread = threading.Thread(target=background_poller, daemon=True)
poller_thread.start()

# -----------------------
# Feature engineering (mirror training pipeline)
# -----------------------
def compute_aggregated_features_from_buffer(buf_deque):
    """
    Given a deque of raw samples for one container (each sample is the dict created above),
    compute the aggregated features expected by the model.
    Returns (feature_dict, debug_dict) or (None, None) if insufficient data.
    """
    if not buf_deque or len(buf_deque) < 3:
        return None, None  # need at least a few samples

    df = pd.DataFrame(list(buf_deque))
    # ensure sorted by time
    df = df.sort_values("timestamp").reset_index(drop=True)

    # CPU series and memory series
    cpu_series = df["cpu_percent"].astype(float).to_numpy()
    mem_series = df["mem_used_mb"].astype(float).to_numpy()
    net_tx_series = df["net_tx_mb"].astype(float).to_numpy()
    net_rx_series = df["net_rx_mb"].astype(float).to_numpy()

    # Basic stats
    cpu_mean = float(np.nanmean(cpu_series))
    cpu_max = float(np.nanmax(cpu_series))
    cpu_std = float(np.nanstd(cpu_series, ddof=0))
    mem_mean = float(np.nanmean(mem_series))
    mem_max = float(np.nanmax(mem_series))
    mem_std = float(np.nanstd(mem_series, ddof=0))
    net_tx_mean = float(np.nanmean(net_tx_series))
    net_rx_mean = float(np.nanmean(net_rx_series))

    # Slopes: use np.gradient on the series (same as training)
    try:
        cpu_slope = float(np.gradient(cpu_series).mean()) if len(cpu_series) >= 2 else 0.0
    except Exception:
        cpu_slope = 0.0
    try:
        mem_slope = float(np.gradient(mem_series).mean()) if len(mem_series) >= 2 else 0.0
    except Exception:
        mem_slope = 0.0

    # EWMA on CPU and memory
    try:
        ser = pd.Series(cpu_series)
        cpu_ewma = float(ser.ewm(span=EWMA_SPAN, adjust=False).mean().iloc[-1])
    except Exception:
        cpu_ewma = cpu_mean
    try:
        serm = pd.Series(mem_series)
        mem_ewma = float(serm.ewm(span=EWMA_SPAN, adjust=False).mean().iloc[-1])
    except Exception:
        mem_ewma = mem_mean

    # Deviations vs baseline (if baseline available we'll compute z later)
    cpu_dev = float(cpu_series[-1] - cpu_ewma)
    mem_dev = float(mem_series[-1] - mem_ewma)

    # Compose feature dict (names must match training's expected columns)
    feat = {
        "cpu_mean": cpu_mean,
        "cpu_max": cpu_max,
        "cpu_std": cpu_std,
        "mem_mean_mb": mem_mean,
        "mem_max_mb": mem_max,
        "mem_std_mb": mem_std,
        "net_tx_mean_mb": net_tx_mean,
        "net_rx_mean_mb": net_rx_mean,
        "cpu_slope": cpu_slope,
        "mem_slope": mem_slope,
        "cpu_dev": cpu_dev,
        "mem_dev": mem_dev,
    }

    debug = {
        "cpu_series": cpu_series.tolist(),
        "mem_series": mem_series.tolist(),
        "cpu_ewma": cpu_ewma,
        "mem_ewma": mem_ewma,
        "latest_sample": df.iloc[-1].to_dict()
    }

    return feat, debug

# -----------------------
# Helper: compute z-scores relative to baseline table (if present)
# -----------------------
def inject_baseline_zscores(container_name, features):
    """
    features: dict (cpu_mean, mem_mean_mb, cpu_dev, mem_dev, etc.)
    Mutates features by adding cpu_z and mem_z fields if baseline exists.
    """
    if baseline_df is None:
        # not available -> fallback to 0 z-scores
        features["cpu_z"] = 0.0
        features["mem_z"] = 0.0
        return features

    if container_name not in baseline_df.index:
        features["cpu_z"] = 0.0
        features["mem_z"] = 0.0
        return features

    row = baseline_df.loc[container_name]
    # baseline columns expected: cpu_mean_container, cpu_std_container, mem_mean_container, mem_std_container
    # try multiple naming variants
    cpu_mean_col = None
    cpu_std_col = None
    mem_mean_col = None
    mem_std_col = None

    for cand in ["cpu_mean_container", "cpu_mean", "cpu_mean_train"]:
        if cand in baseline_df.columns:
            cpu_mean_col = cand
            break
    for cand in ["cpu_std_container", "cpu_std", "cpu_std_train"]:
        if cand in baseline_df.columns:
            cpu_std_col = cand
            break
    for cand in ["mem_mean_container", "mem_mean", "mem_mean_train"]:
        if cand in baseline_df.columns:
            mem_mean_col = cand
            break
    for cand in ["mem_std_container", "mem_std", "mem_std_train"]:
        if cand in baseline_df.columns:
            mem_std_col = cand
            break

    try:
        cpu_mean_base = float(row[cpu_mean_col]) if cpu_mean_col else 0.0
        cpu_std_base = float(row[cpu_std_col]) if cpu_std_col else 1.0
        mem_mean_base = float(row[mem_mean_col]) if mem_mean_col else 0.0
        mem_std_base = float(row[mem_std_col]) if mem_std_col else 1.0
    except Exception:
        cpu_mean_base, cpu_std_base, mem_mean_base, mem_std_base = 0.0, 1.0, 0.0, 1.0

    # avoid division by zero
    cpu_std_base = cpu_std_base if cpu_std_base > 1e-6 else 1e-6
    mem_std_base = mem_std_base if mem_std_base > 1e-6 else 1e-6

    features["cpu_z"] = (features["cpu_mean"] - cpu_mean_base) / cpu_std_base
    features["mem_z"] = (features["mem_mean_mb"] - mem_mean_base) / mem_std_base
    return features

# -----------------------
# API endpoints
# -----------------------
@app.route("/api/live-containers", methods=["GET"])
def api_live_containers():
    """
    Return a list of running containers available for live inspection.
    """
    if docker_client is None:
        return jsonify([])

    try:
        containers = docker_client.containers.list(all=False)
        out = []
        for c in containers:
            out.append({
                "id": c.id[:12],
                "name": c.name,
                "status": c.status,
                "image": c.attrs.get('Config', {}).get('Image', '')
            })
        return jsonify(out)
    except Exception as e:
        print("⚠️ api_live_containers error:", e)
        return jsonify([]), 500

@app.route("/api/live-prediction/<container_id>", methods=["GET"])
def live_prediction(container_id):
    """
    Compute live prediction for container_id (12-char container id).
    Uses the buffer contents for that container.
    """
    # require model and feature list
    if model is None or feature_cols is None:
        return jsonify({"error": "Model or feature columns not loaded."}), 500

    with buffers_lock:
        buf = buffers.get(container_id)
        if not buf or len(buf) < 3:
            return jsonify({"error": "Not enough live samples yet for this container."}), 404

        feat, debug = compute_aggregated_features_from_buffer(buf)
        if feat is None:
            return jsonify({"error": "Insufficient data to compute features."}), 404

        # inject z-scores against baseline
        container_name = buf[-1]["container_name"]
        feat = inject_baseline_zscores(container_name, feat)

        # ensure feature order and fill missing
        Xdict = {k: float(feat.get(k, 0.0)) for k in feature_cols}

    # scale features
    try:
        X_df = pd.DataFrame([Xdict])
        if scaler is not None:
            X_scaled = scaler.transform(X_df[feature_cols].values)
        else:
            # fallback: standardize by column mean/std from this vector (not ideal)
            X_scaled = X_df[feature_cols].values
    except Exception as e:
        print("⚠️ scaling error:", e)
        return jsonify({"error": "Scaling failure"}), 500

    # model inference
    try:
        score = float(model.decision_function(X_scaled)[0])  # higher = more normal
        pred = int(model.predict(X_scaled)[0])  # 1 normal, -1 anomaly
    except Exception as e:
        print("⚠️ model inference error:", e)
        return jsonify({"error": "Model inference failed"}), 500

    # map to status + simple confidence heuristic
    status = "Anomaly" if pred == -1 else "Normal"
    # confidence: convert decision score to [0,1] roughly (higher absolute score => higher confidence).
    # decision_function is positive for normal; negative for anomalous. We'll map to confidence = 1 - sigmoid(|score|)
    confidence = max(0.0, 1.0 - min(1.0, abs(score)))  # conservative heuristic
    confidence = float(round(confidence, 3))

    # natural language explanation
    mem_msg = ""
    if feat.get("mem_slope", 0) > 0:
        mem_msg = f"Memory rising {round(feat['mem_slope'],3)} units."
    elif feat.get("mem_slope", 0) < 0:
        mem_msg = f"Memory decreasing {round(abs(feat['mem_slope']),3)} units."
    else:
        mem_msg = "Memory stable."

    cpu_msg = ""
    if feat.get("cpu_slope", 0) > 0:
        cpu_msg = f"CPU rising {round(feat['cpu_slope'],3)} units."
    elif feat.get("cpu_slope", 0) < 0:
        cpu_msg = f"CPU decreasing {round(abs(feat['cpu_slope']),3)} units."
    else:
        cpu_msg = "CPU stable."

    explanation = f"{mem_msg} {cpu_msg} (z: cpu {round(feat.get('cpu_z',0),2)}, mem {round(feat.get('mem_z',0),2)})"

    resp = {
        "container_id": container_id,
        "container_name": container_name,
        "status": status,
        "anomaly_score": score,
        "confidence": confidence,
        "explanation": explanation,
        "trend": {
            "cpu_slope": feat.get("cpu_slope", 0),
            "mem_slope": feat.get("mem_slope", 0),
        },
        "features": Xdict,
        "debug": debug
    }
    return jsonify(resp)




@app.route("/api/containers", methods=["GET"])
def api_containers_alias():
    return api_live_containers()

@app.route("/api/predictions/<container_id>", methods=["GET"])
def predictions_alias(container_id):
    return live_prediction(container_id)



# Root: minimal page
@app.route("/")
def home():
    return jsonify({"message": "AetherDock live API running", "live_containers_url": "/api/live-containers"})

# -----------------------
# Run server
# -----------------------

from flask import render_template

@app.route("/dashboard")
def dashboard():
    return render_template("index.html")


if __name__ == "__main__":
    print("✅ Model, scaler, features, and baseline loaded (where available).")
    print("🚀 Starting AetherDock Live Monitor ...")
    app.run(debug=True, host="127.0.0.1", port=5000)


