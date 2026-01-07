Write-Host "🚀 Starting RAG-Ops System Production Setup..." -ForegroundColor Green

# 1. Check for .env
if (-not (Test-Path ".env")) {
    Write-Host "⚠️ .env not found. Creating from example..." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host "Please edit .env with your OpenAI Key and run this script again." -ForegroundColor Red
    exit
}

# 2. Install Dependencies
Write-Host "📦 Installing dependencies with uv..." -ForegroundColor Cyan
uv venv
.venv\Scripts\activate
uv pip install -r pyproject.toml

# 3. Start Infrastructure
Write-Host "🐳 Starting Docker Infrastructure (Milvus, MLflow, Postgres)..." -ForegroundColor Cyan
docker-compose up -d
Write-Host "⏳ Waiting 20s for services to initialize..." -ForegroundColor DarkGray
Start-Sleep -Seconds 20

# 4. Ingest Data (Idempotent check could be added here, but for demo we re-run)
Write-Host "📚 Ingesting Knowledge Base..." -ForegroundColor Cyan
$env:PYTHONPATH="."
python -m src.ingestion.pipeline

# 5. Start API
Write-Host "🔥 Starting FastAPI Server..." -ForegroundColor Green
Write-Host "   Swagger UI: http://localhost:8000/docs"
Write-Host "   MLflow UI:  http://localhost:5000"
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000