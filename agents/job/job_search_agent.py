from langchain.tools import tool
from typing import List, Dict, TypedDict

import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from utils.constants import MODEL_ID

# Load environment variables
load_dotenv()
model_id = MODEL_ID

# Initialize the Gemini agent
llm = ChatGoogleGenerativeAI(model=model_id, temperature=0.0)

@tool
def tavily_search_tool(query: str) -> str:
    """Performs a web search and returns the results."""
    search = TavilySearchResults(max_results=5)
    results = search.invoke(query)
    return results

# 🧠 Prompt Template: Just ask LLM to extract fields and act
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a smart web assistant that helps users find and extract live and relevant information from the internet. For each query, perform the following tasks based on the type of information being requested — job listings, community events, or mentorship programs:

---

🔹 JOB LISTINGS

1. Search for job openings based on the provided criteria, such as:

    - Required skills
    - Location
    - Experience level

2. Prioritize recently posted jobs (within the last 7–14 days) that are actively accepting applications.

    - Look for signs like "Apply Now", live application links, or posting timestamps.

3. Filter out expired or inactive job listings.

4. Extract the following fields:

    - Job Title  
    - Company Name  
    - Location  
    - Salary (if available)  
    - Experience required  
    - Posting Date  
    - Job description  
    - Direct link to apply

---

🔹 COMMUNITY EVENTS

1. Look for active community-driven or professional events related to the query.

2. Filter for upcoming or ongoing events — avoid events with past dates.

3. Extract the following fields:

    - Event Name  
    - Event Type (e.g., workshop, meetup, conference)  
    - Location or Online Link  
    - Date & Time  
    - Event Description  
    - Registration or Info Link

---

🔹 MENTORSHIP PROGRAMS

1. Search for structured mentorship programs relevant to the user's query (based on domain, career level, or interest).

2. Prioritize ongoing or enrolling programs.

3. Extract the following fields:

    - Program Name  
    - Mentor Name (if known)  
    - Duration (e.g., 3 months, 12 weeks)  
    - Location or Online  
    - Program Description  
    - Signup or Application Link

---

Your goal is to extract **structured and relevant information**, even if only partial data is available. If context is ambiguous, include the available info and flag incomplete results.

Return the final result as a list of structured objects containing the type of result (job, event, or mentorship) and the relevant fields.

                """),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 🧰 Define tools
tools = [tavily_search_tool]

# 🎯 Create the agent
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 🌐 LangGraph State Definition
class GraphState(TypedDict):
    input: str
    output: str
    # search_results: any

workflow = StateGraph(GraphState)

# 🧩 Agent node
def call_agent(state):
    print(f"\n🟡 User Query: {state['input']}\n")
    result = executor.invoke({"input": state["input"]})
    return {"output": result.get("output", str(result))}

# def web_search(state):
#     print(f"\n🔍 Web Search: {state['input']}\n")
#     results = web_agent.invoke({"input":state["input"]})
#     return {"search_results": results["output"]}
workflow.add_node("agent", call_agent)
# workflow.add_node("web_search", web_search)
workflow.set_entry_point("agent")
# workflow.add_edge("agent", "web_search")
# workflow.add_edge("web_search", END)
web_agent = workflow.compile()