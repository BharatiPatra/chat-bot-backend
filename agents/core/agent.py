import json
from agents.job.job_search_agent import web_agent
from agents.feature.feature_agent import feature_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START
# from langgraph.prebuilt import ToolExecutor
from typing import TypedDict, List, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

from utils.constants import MODEL_ID


model_id = MODEL_ID

# Define state
class AgentState(TypedDict):
    user_query: str
    intent: str
    chat_history: List[Dict[str, str]]
    tool_results: dict
    response: str

# Define dummy tools (replace with actual implementations)
def feature_tool(query: str) -> Dict[str, List[str]]:
    result = feature_agent(query)
    return {"feature_tool": f"'{result}'"}

def web_tool(query: str) -> Dict[str, str]:
    result = web_agent.invoke({"input": query})
    return {"web_search": f"'{result['output']}'." }#, "search_results": result["search_results"]}

# Assume you have an instance of a Langchain Chat Model (e.g., ChatOpenAI)
# Replace with your actual LLM setup
llm = ChatGoogleGenerativeAI(model=model_id, temperature=0.0)

# extract_intent = create_intent_classifier(llm)

# Define other nodes
def feature(state: AgentState) -> Dict[str, Dict]:
    result = feature_tool(state["user_query"])
    # Ensure tool_results is initialized
    if "tool_results" not in state or state["tool_results"] is None:
        state["tool_results"] = {}

    # Merge/accumulate results
    state["tool_results"].update(result)

    return state
    # return {"tool_results": result}

def web(state: AgentState) -> Dict[str, Dict]:
    result = web_tool(state["user_query"])
    if "tool_results" not in state or state["tool_results"] is None:
        state["tool_results"] = {}

    # Merge/accumulate results
    state["tool_results"].update(result)
    return state

def generate_response(state: AgentState) -> Dict[str, str]:
    prompt = ChatPromptTemplate.from_messages([
    ("system", 
 "You are a friendly and knowledgeable job search assistant. Always provide clear, helpful, and well-formatted answers to the user's questions.\n\n"
 "1. **Never** mention tools, internal steps, or processes — the user only cares about the final results.\n"
 "2. If job listings, community information, or any other content that can benefit from structured formatting is available, present it in a clean, readable format using markdown. For example:\n\n"
 "**Job Title:** Java Developer  \n"
 "**Company:** Acme Corp  \n"
 "**Location:** New York, NY  \n"
 "**Summary:** A mid-level Java developer role requiring 3+ years of experience.\n\n"
 "3. After presenting the results, provide helpful explanations or next-step suggestions if applicable.\n"
 "4. If the results are too generic or insufficient, guide the user to refine their question with specifics like location, salary range, or experience level.\n"
 "5. Some information has been collected from external sources and is included under the 'Additional information' section — use this data as well when forming your response, without mentioning how it was collected."
),
("user", 
 "User question: {user_query}\n\n"
 "Tools output:\n{tools_output}\n\n"
 )
 ])
    tool_results = state.get("tool_results") or {}

    if tool_results:
        tool_results_str = "\n\n".join(
            f"[{tool}]\n{json.dumps(result, indent=2)}"
            for tool, result in tool_results.items()
        )
    else:
        tool_results_str = "No tool results available."

    formated_prompt = prompt.format_messages(
        user_query=state["user_query"],
        tools_output=tool_results_str,
        # additional_info=info
    )
    llm_response = llm.invoke(formated_prompt)
    return {"response": llm_response.content}

# Construct graph
workflow = StateGraph(AgentState)

# Add nodes for each stage
workflow.add_node("feature", feature)
workflow.add_node("web", web)
workflow.add_node("generate_response", generate_response)


# Wire up the edges of the graph
workflow.add_edge(START, "feature")
workflow.add_edge("feature", "web")
workflow.add_edge("web", "generate_response")
workflow.add_edge("generate_response", END)

# Compile the workflow
core_agent = workflow.compile()

if __name__ == "__main__":
    # Example usage
    inputs = {
        "user_query": "Tell me about the latest community events",
        "chat_history": [
            {"role": "human", "content": "Hi there"},
            {"role": "ai", "content": "Hello! How can I help you?"},
            {"role": "human", "content": "Tell me about available jobs"},
            {"role": "ai", "content": "We have job openings in software and data science."}
        ],
        "tool_results": {},  # optional but safe to include
        "intent": "",        # will be set by extract_intent
        "response": "",      # will be set by generate_response
    }
    for output in core_agent.astream(inputs):
        for key, value in output.items():
            print(f"Node '{key}':")
            print(value)