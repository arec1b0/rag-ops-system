from langgraph.graph import END, StateGraph
from src.agent.state import AgentState
from src.agent.nodes import retrieve, grade_documents, generate

# 1. Initialize Graph
workflow = StateGraph(AgentState)

# 2. Add Nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

# 3. Add Edges
# Entry point -> Retrieve
workflow.set_entry_point("retrieve")

# Retrieve -> Grade
workflow.add_edge("retrieve", "grade_documents")

# Grade -> Generate
# In a full agentic loop, we would add a conditional edge here:
# If no docs are relevant -> Rewrite query.
# For this step, we proceed linearly to Generation to ensure connectivity.
workflow.add_edge("grade_documents", "generate")

# Generate -> End
workflow.add_edge("generate", END)

# 4. Compile
app = workflow.compile()