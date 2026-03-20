#!/bin/bash
# =============================================================
#  CMARG Pro — Setup completo para Cloud Run + BigQuery
#  Ejecutar este script desde tu terminal (no desde IDE)
# =============================================================

set -e

echo "=========================================="
echo "  CMARG Pro — Setup GCP + BigQuery"
echo "=========================================="

PROJECT_ID="geologgia-map"
REGION="us-central1"
SERVICE_NAME="cmarg-dashboard"

# 1. Configurar proyecto
echo ""
echo "📎 Configurando proyecto: $PROJECT_ID"
gcloud config set project $PROJECT_ID

# 2. Habilitar APIs necesarias
echo ""
echo "🔌 Habilitando APIs..."
gcloud services enable \
    bigquery.googleapis.com \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    --project=$PROJECT_ID

echo "✅ APIs habilitadas"

# 3. Instalar dependencias Python (si no están)
echo ""
echo "📦 Instalando dependencias Python..."
pip install google-cloud-bigquery db-dtypes tqdm pandas 2>/dev/null || \
pip3 install google-cloud-bigquery db-dtypes tqdm pandas

# 4. Crear dataset y subir datos a BigQuery
echo ""
echo "📊 Ejecutando migración de datos a BigQuery..."
cd /Users/cavila/Downloads/CMARG
python3 scripts/upload_to_bigquery.py

# 5. Build y deploy a Cloud Run
echo ""
echo "=========================================="
echo "  DEPLOY a Cloud Run"
echo "=========================================="
echo ""

read -p "¿Deseas deployar a Cloud Run ahora? [s/N]: " deploy
if [[ "$deploy" =~ ^[sS]$ ]]; then
    echo "🐳 Building y deploying..."
    gcloud run deploy $SERVICE_NAME \
        --source . \
        --region $REGION \
        --platform managed \
        --memory 2Gi \
        --cpu 2 \
        --timeout 300 \
        --min-instances 0 \
        --max-instances 3 \
        --set-env-vars "USE_BIGQUERY=true,GCP_PROJECT_ID=$PROJECT_ID,BQ_DATASET=cmarg" \
        --allow-unauthenticated \
        --port 8080
    
    echo ""
    echo "🎉 ¡Deploy completado!"
    echo "URL: $(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')"
else
    echo "⏭️ Deploy saltado. Puedes deployar luego con:"
    echo "  gcloud run deploy cmarg-dashboard --source . --region $REGION --platform managed --memory 2Gi --cpu 2 --timeout 300 --set-env-vars 'USE_BIGQUERY=true,GCP_PROJECT_ID=$PROJECT_ID,BQ_DATASET=cmarg' --allow-unauthenticated --port 8080"
fi

echo ""
echo "=========================================="
echo "  ¡Setup completado!"
echo "=========================================="
