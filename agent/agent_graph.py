from langgraph.graph import StateGraph, END
from agent.agents import AgentState, run_history_summary, run_query_rag, run_response_summarize

builder = StateGraph(AgentState)

# Add nodes
builder.add_node("history_summary", run_history_summary)
builder.add_node("query_rag", run_query_rag)
builder.add_node("response_summarize", run_response_summarize)

# Set up transitions
builder.set_entry_point("history_summary")
builder.add_edge("history_summary", "query_rag")
builder.add_edge("query_rag", "response_summarize")
builder.add_edge("response_summarize", END)

# Step 3: Compile the graph
agent_graph = builder.compile()