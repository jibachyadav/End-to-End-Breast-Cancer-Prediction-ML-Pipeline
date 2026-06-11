#!/bin/bash
set -e

echo "Starting Breast Cancer Prediction API."

# Wait for MariaDB to be ready
echo "Waiting for MariaDB."
while ! mysqladmin ping -h mariadb -u bc_user -pbc_password123 --silent; do
    sleep 2
done
echo "MariaDB is ready!"

# Run pipeline if no model exists
if [ ! -f "artifacts/models/best_model.pkl" ]; then
    echo "No model found — running training pipeline."
    python src/pipeline/training_pipeline.py
    echo "Training complete!"
else
    echo "Model already exists — skipping training"
fi

# Start FastAPI
echo "Starting FastAPI."
exec uvicorn api.main:app --host 0.0.0.0 --port 8000