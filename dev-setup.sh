#!/bin/bash
# Quickstart script for development

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🏭 PREDICTIVE MAINTENANCE AI - DEVELOPMENT SETUP            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Create directories
echo "📁 Creating directories..."
mkdir -p logs models data/raw data/processed

# Copy environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Created .env file (please review settings)"
fi

echo ""
echo "✅ Development setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run training: python -m pipelines.complete_pipeline"
echo "3. Start API: python -m uvicorn api.main:app --reload"
echo "4. Start dashboard (new terminal): streamlit run dashboard/app.py"
echo ""
