# Predictive Maintenance AI

Machine learning system for predicting industrial equipment failures before they occur. Uses time-series sensor data to forecast Remaining Useful Life (RUL) and failure probability.

## Features

- **Failure Prediction**: Binary classification for imminent failures
- **RUL Estimation**: Regression model for remaining useful life in cycles
- **Feature Engineering**: 364 engineered features from 21 sensors
- **Multiple Models**: XGBoost, LightGBM, Random Forest, Ridge Regression
- **REST API**: FastAPI server with prediction endpoints
- **Dashboard**: Streamlit web interface for monitoring
- **MLflow Tracking**: Experiment tracking and model registry

## Quick Start

### Local Setup

```bash
# 1. Clone and setup
git clone https://github.com/csaw5866/predictive-maintenance-ai.git
cd predictive-maintenance-ai

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env

# 5. Train models (first time only)
python -m pipelines.train

# 6. Start API server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8003

# 7. Start dashboard (in new terminal)
source venv/bin/activate
streamlit run dashboard/app.py --server.port 8501
```

Access:
- **API Documentation**: http://localhost:8003/docs
- **Dashboard**: http://localhost:8501
- **Health Check**: http://localhost:8003/health

### Docker Setup

```bash
# Build and start all services
docker compose up --build -d

# View logs
docker compose logs -f api
docker compose logs -f dashboard

# Stop all services
docker compose down
```

Access (same URLs as local setup).

## Project Structure

```
predictive-maintenance-ai/
├── api/                        # FastAPI REST API
│   └── main.py                # API endpoints and model serving
├── dashboard/                  # Streamlit dashboard
│   └── app.py                 # Web UI for monitoring
├── pipelines/                  # Training pipelines
│   └── train.py               # End-to-end training pipeline
├── pma/                        # Core ML package
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── logger.py              # Logging setup
│   ├── data.py                # Data loading & preprocessing
│   ├── features.py            # Feature engineering
│   ├── models.py              # Model training & evaluation
│   └── schemas.py             # Pydantic data models
├── docker/                     # Dockerfiles
│   ├── Dockerfile.api
│   └── Dockerfile.dashboard
├── tests/                      # Unit tests
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
├── data/                       # Datasets
│   ├── raw/                   # NASA C-MAPSS raw data
│   └── processed/             # Feature-engineered data
├── models/                     # Trained models
│   ├── best_classifier.pkl    # Failure prediction model
│   ├── best_regressor.pkl     # RUL estimation model
│   └── features.json          # Feature metadata
├── docker-compose.yml          # Multi-service orchestration
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project metadata
├── .env.example               # Environment template
└── README.md                  # This file
```

## API Endpoints

### Health Check
```bash
GET /health
```
Returns API status and loaded models count.

### Predict Failure
```bash
POST /predict/failure
Content-Type: application/json

{
  "features": [100.5, 102.3, 99.8, ...]  # 364 features
}
```
Returns failure probability (0.0-1.0) and classification.

### Predict RUL
```bash
POST /predict/rul
Content-Type: application/json

{
  "features": [100.5, 102.3, 99.8, ...]  # 364 features
}
```
Returns estimated remaining useful life in cycles.

### Example Usage
```bash
# Health check
curl http://localhost:8003/health

# Predict failure (example with random features)
curl -X POST http://localhost:8003/predict/failure \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, ...]}'
```

## ML Models & Performance

Trained on **NASA C-MAPSS Turbofan Engine Degradation Dataset**:
- **Training samples**: 20,631
- **Sensors**: 21 sensor readings per cycle
- **Features**: 364 engineered features
- **Operational settings**: 3 parameters

### Model Performance

| Task | Model | Metric | Value |
|------|-------|--------|-------|
| Failure Classification | XGBoost | ROC-AUC | 0.4844 |
| Failure Classification | XGBoost | F1-Score | 0.0128 |
| RUL Regression | Ridge | RMSE | 66.72 |
| RUL Regression | Ridge | MAE | 55.32 |

**Note**: The dataset is imbalanced (failures are rare), leading to lower F1 scores. ROC-AUC and RMSE are more relevant metrics for this use case.

## Configuration

Edit `.env` file or set environment variables:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8003

# Paths
MODELS_PATH=./models
DATA_PATH=./data
MLFLOW_TRACKING_URI=file:./mlruns

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Model Training
RANDOM_STATE=42
TEST_SIZE=0.2
```

## Development

### Retrain Models

```bash
# Run complete training pipeline
python -m pipelines.train

# Models will be saved to ./models/
# - best_classifier.pkl
# - best_regressor.pkl
# - features.json (metadata)
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=pma --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

### Code Quality

```bash
# Format code
black pma/ api/ dashboard/ pipelines/ tests/

# Lint
flake8 pma/ api/ dashboard/ --max-line-length=120

# Type checking
mypy pma/ api/ dashboard/ --ignore-missing-imports
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| ML Framework | scikit-learn, XGBoost, LightGBM |
| API Framework | FastAPI, Uvicorn |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Streamlit |
| Model Tracking | MLflow |
| Containerization | Docker, Docker Compose |
| Testing | Pytest |
| Code Quality | Black, Flake8, Mypy |

## Dataset

**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**

The dataset simulates turbofan engine degradation with:
- 100 engines
- 21 sensor measurements per engine cycle
- 3 operational settings
- Run-to-failure data (each engine runs until failure)

**Citation**:
```bibtex
@article{saxena2008damage,
  title={Damage propagation modeling for aircraft engine run-to-failure simulation},
  author={Saxena, Abhinav and Goebel, Kai and Simon, Don and Eklund, Neil},
  journal={2008 International Conference on Prognostics and Health Management},
  pages={1--9},
  year={2008},
  organization={IEEE}
}
```

**Source**: [NASA Prognostics Center of Excellence](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

## Troubleshooting

### Port Already in Use
```bash
# Kill process on port 8003
lsof -ti:8003 | xargs kill -9

# Or use different port
python -m uvicorn api.main:app --port 8004
```

### Models Not Found
```bash
# Train models first
python -m pipelines.train

# Check models directory
ls -l models/
# Should contain: best_classifier.pkl, best_regressor.pkl, features.json
```

### Docker Build Fails
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker compose build --no-cache
```

### Import Errors
```bash
# Ensure you're in virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pma; print('OK')"
```

### Dashboard Won't Load
```bash
# Check if API is running
curl http://localhost:8003/health

# Ensure Streamlit port is available
lsof -ti:8501 | xargs kill -9

# Restart dashboard
streamlit run dashboard/app.py --server.port 8501
```

## Contributing

Contributions welcome! Areas for improvement:
- [ ] Add LSTM/Transformer models for temporal patterns
- [ ] Implement real-time data ingestion pipeline
- [ ] Add authentication to API endpoints
- [ ] Expand test coverage
- [ ] Add data drift detection
- [ ] Implement model explainability (SHAP)
- [ ] Add CI/CD pipeline
- [ ] Kubernetes deployment manifests

## License

MIT License

Copyright (c) 2026 Joey Hieronimy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
