from agents.job.job_search_agent import job_agent
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
    return {"results": [f"Found product/feature related to query - '{result}'"]}

def job_tool(query: str) -> Dict[str, str]:
    result = job_agent.invoke({"input": query})
    return {"results": f"Answer from Job search - '{result['output']}'.", "search_results": result["search_results"]}

def profile_update_tool(user_id: str, updates: Dict[str, Any]) -> Dict[str, bool]:
    return {"results": True}

# --- Intent Extraction using a Language Model ---
def create_intent_classifier(llm: BaseChatModel):
    prompt = ChatPromptTemplate.from_messages([
        ("system", '''
        You are an intelligent assistant designed to classify user queries based on their intent. Always respond with only one of the following intents:

            1. feature: When the user is exploring features, offerings or general questions or seeks help on common topics (e.g., how-to, what is, where to find, troubleshooting).
            2. job: When the user asks related to jobs, events, or programs.
            3. profile_update: When the user wants to change or manage their personal profile information(ex: skill, experience, location etc.) or expresses interest in registering, creating an account, or signing up.
            4. unknown: When the user asks a question that does not fit into any of the above categories.

            Carefully analyze the user's query and return the most suitable intent as a single lowercase string (e.g., job).
        '''),
        # ("human", "{user_query}"),
    ])
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    def extract_intent(state: AgentState) -> Dict[str, str]:
        try:
            user_query = state["user_query"]
            history = state.get("chat_history", [])
            # Reconstruct messages from history
            message_chain = [
                prompt.append(HumanMessage(content=msg["content"])) if msg["role"] == "human"
                else prompt.append(AIMessage(content=msg["content"]))
                for msg in history
            ]
            # Add current user query at the end
            prompt.append(HumanMessage(content=user_query))
            # print(f"Message Chain: {prompt.format_messages(user_query=message_chain)}")
            # Now use the full list of messages
            formatted_messages = prompt.format()
            llm_response = llm.invoke(formatted_messages)
            intent = output_parser.invoke(llm_response)
            # intent = chain.invoke({"user_query":message_chain}).content.strip().lower()

            # intent = chain.invoke({"user_query": state["user_query"]}).strip().lower()

            if intent in ["feature", "job", "profile_update", "unknown"]:
                return {"intent": intent}
            else:
                return {"intent": "unknown"}
        except Exception as e:
            print(f"Error classifying intent: {e}")
            return {"intent": "unknown"}

    return extract_intent
# --- End of Intent Extraction ---

# Assume you have an instance of a Langchain Chat Model (e.g., ChatOpenAI)
# Replace with your actual LLM setup
llm = ChatGoogleGenerativeAI(model=model_id, temperature=0.0)

extract_intent = create_intent_classifier(llm)

# Define other nodes
def feature(state: AgentState) -> Dict[str, Dict]:
    result = feature_tool(state["user_query"])
    return {"tool_results": result}

def job(state: AgentState) -> Dict[str, Dict]:
    result = job_tool(state["user_query"])
    return {"tool_results": result}

def profile_update(state: AgentState) -> Dict[str, Dict]:
    result = profile_update_tool("user123", {"update_info": state["user_query"]})
    return {"tool_results": result}

def unknown_query(state: AgentState) -> Dict[str, Dict]:
    return {"tool_results": "Please ask questions related to career or jobs"}

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
#  "Additional information:\n{additional_info}\n"
 )
 ])
    info = state["tool_results"]["search_results"] 
    formated_prompt = prompt.format_messages(
        user_query=state["user_query"],
        tools_output=state.get("tool_results", {}).get("results", ""),
        # additional_info=info
    )
    llm_response = llm.invoke(formated_prompt)
    return {"response": llm_response.content}

# Construct graph
workflow = StateGraph(AgentState)

# Add nodes for each stage
workflow.add_node("extract_intent", extract_intent)
workflow.add_node("feature", feature)
workflow.add_node("job", job)
workflow.add_node("profile_update", profile_update)
workflow.add_node("unknown", unknown_query)
workflow.add_node("generate_response", generate_response)

# Define routing logic based on classified intent
def condition(state: AgentState):
    intent = state.get("intent", "").strip().lower()
    valid_intents = ["feature", "job", "profile_update", "unknown"]
    if intent in valid_intents:
        return intent
    return "unknown"


# Wire up the edges of the graph
workflow.add_edge(START, "extract_intent")
workflow.add_conditional_edges("extract_intent", condition)
workflow.add_edge("feature", "generate_response")
workflow.add_edge("job", "generate_response")
workflow.add_edge("profile_update", "generate_response")
workflow.add_edge("unknown", "generate_response")
workflow.add_edge("generate_response", END)

# Compile the workflow
core_agent = workflow.compile()


# Example usage
# inputs = {"user_query": "Tell me about the latest community events"}
# chat_history = [
#     {"role": "human", "content": "Hi there"},
#     {"role": "ai", "content": "Hello! How can I help you?"},
#     {"role": "human", "content": "Tell me about available jobs"},
#     {"role": "ai", "content": "We have job openings in software and data science."}
# ]
# new_query = "Can you check java job openings?"

# inputs = {
#     "user_query": new_query,
#     "chat_history": chat_history,
#     "tool_results": {},  # optional but safe to include
#     "intent": "",        # will be set by extract_intent
#     "response": "",      # will be set by generate_response
# }
# for output in core_agent.astream(inputs):
#     for key, value in output.items():
#         print(f"Node '{key}':")
#         print(value)
