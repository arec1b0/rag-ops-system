from typing import Any, Dict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from src.core.config import settings
from src.agent.state import AgentState
from src.agent.tools import retriever_tool

# Initialize LLM
# We use temperature=0 for deterministic outputs in logic nodes (grading)
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0, 
    api_key=settings.OPENAI_API_KEY
)

# --- DATA MODELS ---

class Grade(BaseModel):
    """Binary score for relevance check."""
    score: Literal["yes", "no"] = Field(
        description="Relevance score 'yes' or 'no'"
    )

# --- PROMPTS FOR A/B TESTING ---

PROMPT_A = """You are a helpful assistant. Use the context to answer the question.
Context: {context}
Question: {question}
Answer:"""

PROMPT_B = """You are a Senior MLOps Engineer named 'Dani-Bot'. 
You speak in a professional, direct, and technical tone. 
Use the retrieved context to provide a concise, production-ready answer.
If the context is missing, admit it immediately.

Context: {context}
Question: {question}
Answer:"""

# --- NODES ---

def retrieve(state: AgentState) -> Dict[str, Any]:
    """
    Node: Retrieves documents based on the question.
    """
    print("---RETRIEVE---")
    question = state["question"]
    # In a real scenario, we would use the vector store retriever here
    # For now, we invoke the tool directly
    documents = [retriever_tool.invoke(question)]
    return {"documents": documents, "question": question}

def generate(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Node: Generates an answer using the retrieved documents and selected Strategy.
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # Read strategy from config (default to 'A')
    strategy = config.get("configurable", {}).get("strategy", "A")
    print(f"--- USING STRATEGY: {strategy} ---")
    
    # Select Template
    template = PROMPT_B if strategy == "B" else PROMPT_A
    
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = prompt | llm | StrOutputParser()
    
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"generation": generation}

def grade_documents(state: AgentState) -> Dict[str, Any]:
    """
    Node: Determines whether the retrieved documents are relevant to the question.
    """
    print("---CHECK RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    
    # Score each doc
    filtered_docs = []
    
    # Prompt for grading
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    
    # Use Pydantic model for structured output
    grader_llm = llm.with_structured_output(Grade)
    
    grader_chain = grade_prompt | grader_llm
    
    for d in documents:
        # grader_chain now returns a Grade object (Pydantic model)
        grade_result = grader_chain.invoke({"question": question, "document": d})
        score = grade_result.score
        
        if score == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            
    return {"documents": filtered_docs, "question": question}