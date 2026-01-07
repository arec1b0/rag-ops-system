from typing import TypedDict, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    Represents the state of the agent graph.
    
    Attributes:
        question (str): The user's input question.
        generation (str): The LLM's generated response.
        documents (List[str]): Retrieved documents context.
    """
    question: str
    generation: str
    documents: List[str]