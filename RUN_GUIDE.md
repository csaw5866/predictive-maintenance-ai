# Complete Shutdown & Restart Guide

## Step 1: Shutdown Everything

```bash
# Kill all running services
pkill -f streamlit          # Stop dashboard
pkill -f uvicorn            # Stop API
pkill -f mlflow             # Stop MLflow (if running)

# Verify everything is stopped
ps aux | grep -E "streamlit|uvicorn|mlflow" | grep -v grep
# Should show no output if all killed
```

## Step 2: Clean Up (Optional - only if you want fresh logs)

```bash
cd /Users/joeyhieronimy/Documents/Projects/predictive-maintenance-ai

# Remove old logs
rm -f /tmp/pma_api.log /tmp/pma_dashboard.log /tmp/dashboard.log

# Clear old MLflow runs (optional - keeps experiment history)
rm -rf ./mlruns
```

## Step 3: Restart the System

### 3a. Activate Virtual Environment
```bash
cd /Users/joeyhieronimy/Documents/Projects/predictive-maintenance-ai
source .venv/bin/activate
```

### 3b. Start API Server (Port 8003)
```bash
# In terminal 1:
cd /Users/joeyhieronimy/Documents/Projects/predictive-maintenance-ai
PYTHONPATH=$PWD MODELS_PATH=$PWD/models \
  .venv/bin/python -m uvicorn api.main:app \
  --host 127.0.0.1 --port 8003 --workers 2

# You should see:
# INFO:     Uvicorn running on http://127.0.0.1:8003
```

### 3c. Start Dashboard (Port 8501)
```bash
# In terminal 2:
cd /Users/joeyhieronimy/Documents/Projects/predictive-maintenance-ai
STREAMLIT_SERVER_HEADLESS=true \
  .venv/bin/streamlit run dashboard/app.py \
  --server.port 8501 --server.address 0.0.0.0

# You should see:
# Streamlit app is running at http://0.0.0.0:8501
```

### 3d. Start MLflow UI (Optional, Port 5000)
```bash
# In terminal 3 (optional for viewing experiments):
cd /Users/joeyhieronimy/Documents/Projects/predictive-maintenance-ai
.venv/bin/mlflow ui --backend-store-uri ./mlruns -p 5000

# You should see:
# [127.0.0.1:5000]
```

## Step 4: Access the Services

Once all three are running:

| Service      | URL                    | Purpose                        |
|--------------|------------------------|--------------------------------|
| Dashboard    | http://localhost:8501  | Machine health & predictions   |
| API          | http://localhost:8003  | REST endpoints for inference   |
| API Docs     | http://localhost:8003/docs | Swagger API documentation  |
| MLflow       | http://localhost:5000  | Training experiment tracking   |

## Quick Restart Script

Create `~/restart_pma.sh`:

```bash
#!/bin/bash
set -e

cd /Users/joeyhieronimy/Documents/Projects/predictive-maintenance-ai

echo "Killing old processes..."
pkill -f streamlit || true
pkill -f uvicorn || true
sleep 2

echo "Starting API..."
PYTHONPATH=$PWD MODELS_PATH=$PWD/models \
  .venv/bin/python -m uvicorn api.main:app \
  --host 127.0.0.1 --port 8003 --workers 2 > /tmp/api.log 2>&1 &

sleep 3

echo "Starting Dashboard..."
STREAMLIT_SERVER_HEADLESS=true \
  .venv/bin/streamlit run dashboard/app.py \
  --server.port 8501 --server.address 0.0.0.0 > /tmp/dashboard.log 2>&1 &

sleep 5

echo "✅ All services started!"
echo "📊 Dashboard: http://localhost:8501"
echo "🔌 API: http://localhost:8003"
echo "📈 API Docs: http://localhost:8003/docs"
```

Then run it:
```bash
chmod +x ~/restart_pma.sh
~/restart_pma.sh
```

## Troubleshooting

**Port already in use:**
```bash
lsof -i :8501  # Check what's on port 8501
lsof -i :8003  # Check what's on port 8003
kill -9 <PID>  # Kill the process
```

**Can't access dashboard:**
- Use `http://localhost:8501` (not `127.0.0.1`)
- Or find your IP: `ipconfig getifaddr en0`
- Then use `http://<YOUR_IP>:8501`

**Models not loading:**
- Check models exist: `ls -la models/best_*.pkl`
- Check paths: `echo $MODELS_PATH`
