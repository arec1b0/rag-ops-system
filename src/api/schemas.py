from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    """
    Request model for the chat endpoint.
    """
    question: str
    thread_id: Optional[str] = None # For future session management

class QueryResponse(BaseModel):
    """
    Response model returning the answer and used sources.
    """
    answer: str
    documents: List[str]