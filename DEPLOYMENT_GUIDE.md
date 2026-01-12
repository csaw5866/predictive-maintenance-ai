# DEPLOYMENT_GUIDE.md

## Deployment Guide: Predictive Maintenance AI Platform

This guide covers deploying the Predictive Maintenance AI platform to production environments.

---

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Cloud Platforms](#cloud-platforms)
5. [Monitoring & Observability](#monitoring--observability)
6. [Troubleshooting](#troubleshooting)

---

## Local Development

### Prerequisites

- Python 3.11+
- PostgreSQL 14+ (optional, can use SQLite)
- Git

### Quick Setup

```bash
# Clone repository
git clone https://github.com/csaw5866/predictive-maintenance-ai.git
cd predictive-maintenance-ai

# Run setup script
bash dev-setup.sh

# Or manually:
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env as needed

# Run training pipeline
python -m pipelines.complete_pipeline

# In separate terminals:
# Terminal 1: MLflow
mlflow server --backend-store-uri ./mlruns --port 5000

# Terminal 2: API
python -m uvicorn api.main:app --reload --port 8000

# Terminal 3: Dashboard
streamlit run dashboard/app.py --server.port 8501
```

---

## Docker Deployment

### Single Machine (Docker Compose)

**Recommended for development and small deployments.**

```bash
# Clone and navigate
git clone https://github.com/csaw5866/predictive-maintenance-ai.git
cd predictive-maintenance-ai

# Configure environment
cp .env.example .env
# Edit .env for your environment

# Build images
docker compose build

# Start services
docker compose up -d

# View logs
docker compose logs -f

# Access services
# Dashboard: http://localhost:8501
# API: http://localhost:8000/docs
# MLflow: http://localhost:5000
# PostgreSQL: localhost:5432

# Stop services
docker compose down

# Clean up (including volumes)
docker compose down -v
```

### Production Docker Setup

For production, create an override compose file:

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  postgres:
    restart: always
    environment:
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}  # Use strong password

  api:
    restart: always
    environment:
      DEBUG: "false"
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G

  dashboard:
    restart: always
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 1G

  mlflow:
    restart: always
```

Deploy with:
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.21+)
- kubectl configured
- Helm 3+

### Helm Chart Setup

Create `helm/pma-chart/Chart.yaml`:

```yaml
apiVersion: v2
name: predictive-maintenance-ai
description: Predictive Maintenance AI Platform
type: application
version: 1.0.0
appVersion: "1.0.0"
```

### Deployment Steps

```bash
# Create namespace
kubectl create namespace pma

# Create ConfigMap for environment
kubectl create configmap pma-config \
  --from-env-file=.env \
  -n pma

# Create secret for sensitive data
kubectl create secret generic pma-secrets \
  --from-literal=db-password=$(openssl rand -base64 32) \
  -n pma

# Install Helm chart
helm install pma ./helm/pma-chart -n pma

# Check status
kubectl get pods -n pma
kubectl logs -f deployment/pma-api -n pma

# Port forward for testing
kubectl port-forward svc/pma-api 8000:8000 -n pma
```

### Example Kubernetes Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pma-api
  namespace: pma
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pma-api
  template:
    metadata:
      labels:
        app: pma-api
    spec:
      containers:
      - name: api
        image: predictive-maintenance-ai:api-latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: pma-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: pma-api
  namespace: pma
spec:
  type: LoadBalancer
  selector:
    app: pma-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

---

## Cloud Platforms

### AWS ECS (Elastic Container Service)

```bash
# Create ECR repository
aws ecr create-repository --repository-name pma-api

# Build and push image
docker build -t pma-api -f docker/Dockerfile.api .
docker tag pma-api:latest ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/pma-api:latest
docker push ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/pma-api:latest

# Create ECS task definition, service, and cluster via AWS Console or CLI
```

### Google Cloud Run

```bash
# Build image
gcloud builds submit --tag gcr.io/${PROJECT_ID}/pma-api \
  -f docker/Dockerfile.api .

# Deploy
gcloud run deploy pma-api \
  --image gcr.io/${PROJECT_ID}/pma-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 3600
```

### Azure Container Instances

```bash
# Build and push to ACR
az acr build --registry ${ACR_NAME} \
  --image pma-api:latest \
  -f docker/Dockerfile.api .

# Deploy
az container create \
  --resource-group ${RESOURCE_GROUP} \
  --name pma-api \
  --image ${ACR_NAME}.azurecr.io/pma-api:latest \
  --cpu 2 --memory 2 \
  --ports 8000 \
  --registry-login-server ${ACR_NAME}.azurecr.io
```

---

## Monitoring & Observability

### Application Metrics

The platform exposes metrics via:
- `/health` - Health check
- `/metrics` - Application metrics

### Prometheus Integration

```yaml
# docker/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'api'
    static_configs:
      - targets: ['api:8000']

  - job_name: 'mlflow'
    static_configs:
      - targets: ['mlflow:5000']
```

### Logging

Logs are written to `logs/app.log` with rotation:

```python
# In pma/logger.py
logger = setup_logging()
```

For centralized logging, configure log forwarding:

```bash
# Example: Send logs to ELK Stack
# Edit pma/logger.py to add Elasticsearch handler
```

### Grafana Dashboard

Create Grafana dashboard to visualize:
- API response times
- Model prediction accuracy
- Data processing pipeline metrics
- Machine health scores

---

## Troubleshooting

### Container won't start

```bash
# Check logs
docker logs <container-id>

# Verify image built correctly
docker run --rm pma-api:latest /bin/bash

# Check resource limits
docker inspect <container-id> | grep -A 10 Resources
```

### Database connection issues

```bash
# Verify PostgreSQL is running
docker ps | grep postgres

# Test connection
psql postgresql://user:pass@postgres:5432/predictive_maintenance

# Check environment variables
docker inspect <api-container> | grep -i database
```

### Model not loading

```bash
# Verify models directory
ls -la models/
file models/best_classifier.pkl

# Check training pipeline completed
docker logs pma-train

# Restart API after models are saved
docker restart pma-api
```

### Memory issues

```bash
# Increase Docker memory limit
docker update --memory 4g <container-id>

# Or in docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Network connectivity

```bash
# Test container network
docker exec pma-api curl http://postgres:5432

# Check port bindings
docker port <container-id>

# Verify network
docker network ls
docker network inspect pma-network
```

---

## Performance Tuning

### API Server

```python
# docker/Dockerfile.api
CMD ["gunicorn", "api.main:app",
     "--workers", "4",
     "--worker-class", "uvicorn.workers.UvicornWorker",
     "--bind", "0.0.0.0:8000",
     "--timeout", "120"]
```

### Database Optimization

```sql
-- Create indexes on frequently queried columns
CREATE INDEX idx_machine_id ON predictions(machine_id);
CREATE INDEX idx_timestamp ON predictions(timestamp DESC);
CREATE INDEX idx_failure_imminent ON predictions(failure_imminent);
```

### Caching

Consider adding Redis for caching:

```python
# In api/main.py
from redis import Redis

cache = Redis(host="redis", port=6379)

@app.get("/cache-test")
async def cached_endpoint():
    cached = cache.get("key")
    if not cached:
        cached = expensive_computation()
        cache.set("key", cached, ex=3600)
    return cached
```

---

## Scaling Considerations

### Horizontal Scaling

```bash
# Docker Compose (simple load balancer)
docker compose scale api=3

# Kubernetes (automatic)
kubectl autoscale deployment pma-api \
  --min=1 --max=10 \
  -n pma
```

### Database

- Use read replicas for scaling reads
- Connection pooling (PgBouncer)
- Sharding for very large datasets

### Feature Store

- Consider Feast for feature management
- Cache computed features in Redis
- Incremental feature computation

---

## Security Best Practices

1. **Secrets Management**
   - Use sealed-secrets or external vaults
   - Never commit .env files

2. **API Security**
   - Add authentication (JWT)
   - Rate limiting
   - CORS configuration

3. **Database**
   - Use strong passwords
   - Enable SSL connections
   - Regular backups

4. **Container Security**
   - Use non-root users
   - Scan images for vulnerabilities
   - Keep base images updated

---

## Support & Issues

- GitHub Issues: Report bugs
- Discussions: Ask questions
- Documentation: Check README.md

---

## License

MIT License - See LICENSE file
