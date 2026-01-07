import mlflow
from src.core.config import settings

def setup_monitoring():
    """
    Configures MLflow tracing for LangChain.
    """
    print(f"Setting up MLflow tracking at {settings.MLFLOW_TRACKING_URI}")
    
    # 1. Set the tracking URI to point to the local server
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    
    # 2. Set the experiment
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    
    # 3. Enable LangChain Tracing
    # We remove 'log_models' and other args that cause the TypeError.
    # calling autolog() without args enables the tracing system by default.
    mlflow.langchain.autolog()
    
    print("MLflow monitoring enabled.")