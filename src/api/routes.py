import random
import mlflow
from fastapi import APIRouter, HTTPException
from langchain_core.runnables import RunnableConfig
from src.api.schemas import QueryRequest, QueryResponse
from src.agent.graph import app as agent_app

router = APIRouter()

@router.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """
    Endpoint with A/B Testing Logic.
    """
    try:
        # 1. Traffic Split (50/50)
        # In production, hash the user_id to ensure consistency (Sticky Sessions)
        strategy = "B" if random.random() > 0.5 else "A"
        
        # 2. Set MLflow Tags for Analysis
        # This allows you to filter runs by 'strategy=A' vs 'strategy=B' in the dashboard
        mlflow.set_tag("ab_test_strategy", strategy)
        
        # 3. Invoke Graph with Config
        initial_state = {"question": request.question, "documents": []}
        config = RunnableConfig(configurable={"strategy": strategy})
        
        result = await agent_app.ainvoke(initial_state, config=config)
        
        return QueryResponse(
            answer=result["generation"],
            documents=result["documents"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def feedback_endpoint(run_id: str, score: int):
    """
    Capture User Feedback (Thumbs Up/Down).
    score: 1 (Like) or 0 (Dislike)
    """
    # In a real app, write this to Postgres linked to the run_id
    # For this MVP, we log it to MLflow as a metric
    with mlflow.start_run(run_id=run_id, nested=True):
        mlflow.log_metric("user_feedback", score)
    return {"status": "recorded"}