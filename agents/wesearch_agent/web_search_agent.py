from concurrent.futures import ThreadPoolExecutor, as_completed
import os
# Call this after you parse your HTMLs
def extract_all_search_result(infos, query:str, max_workers=5):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extract_info, job, query) for job in infos]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print("Error during LLM call:", e)
    return results

from agents.wesearch_agent.searcher import search_duckduckgo
from agents.wesearch_agent.crawler import crawl_urls
import asyncio

async def run_websearch(query):
    urls = search_duckduckgo(query, max_results=5)
    jobs = await crawl_urls(urls)
    return jobs


from langchain.tools import tool
from typing import List, Dict, TypedDict

import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from utils.constants import MODEL_ID



# Load environment variables
load_dotenv()
model_id = MODEL_ID

# Initialize the Gemini agent
llm = ChatGoogleGenerativeAI(model=model_id, temperature=0.0)

from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class SearchResult(BaseModel):
    type: Literal["job", "event", "mentorship"] = Field(
        description="The type of result. Must be one of: 'job', 'event', or 'mentorship'."
    )

    # Job fields
    title: Optional[str] = Field(
        description="Job title, e.g., 'Senior Python Developer'. Required if type is 'job'."
    )
    company: Optional[str] = Field(
        description="Name of the company offering the job. Required if type is 'job'."
    )
    location: Optional[str] = Field(
        description="Location of the job, event, or mentorship program."
    )
    salary: Optional[str] = Field(
        description="Offered salary for the job, if available. Required if type is 'job'."
    )
    experience_required: Optional[str] = Field(
        description="Years of experience or seniority level required for the job."
    )

    # Event fields
    event_name: Optional[str] = Field(
        description="Name of the event. Required if type is 'event'."
    )
    event_type: Optional[str] = Field(
        description="Type of event (e.g., webinar, workshop, meetup)."
    )
    date: Optional[str] = Field(
        description="Date or datetime of the event."
    )

    # Mentorship fields
    program_name: Optional[str] = Field(
        description="Name of the mentorship program. Required if type is 'mentorship'."
    )
    mentor_name: Optional[str] = Field(
        description="Name of the mentor associated with the program."
    )
    duration: Optional[str] = Field(
        description="Duration of the mentorship program (e.g., '3 months')."
    )

    # Common
    description: Optional[str] = Field(
        description="A detailed description of the job, event, or mentorship program."
    )


class SearchResultList(BaseModel):
    results: List[SearchResult] = Field(
        description="List of extracted search results."
    )

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate



def extract_info(job, query:str):
    prompt = PromptTemplate.from_template("""
    You are an intelligent parser.

    Given a query and an HTML context, extract a list of relevant results.
    Each result can be a job listing, community event, or mentorship program.
    Return only the extracted structured data as JSON using the format instructions below.

    {format_instructions}

    Query: {query}

    Context:
    {context}
    """)
    parser = PydanticOutputParser(pydantic_object=SearchResultList)

    formatted_prompt = prompt.format(context=job["text"], query=query, format_instructions=parser.get_format_instructions())
    response = llm.invoke(formatted_prompt)
    parsed_output = parser.parse(response.content)
    return {"link": job["link"], "out":parsed_output}


# 🌐 LangGraph State Definition
class GraphState(TypedDict):
    input: str
    output: str

workflow = StateGraph(GraphState)

# 🧩 Agent node
async def async_call_agent(state):
    search_result = await run_websearch(state["input"])
    out = extract_all_search_result(search_result, state["input"])
    result = {"output": out}
    return result

def call_agent(state):
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(async_call_agent(state))
    return result

workflow.add_node("agent", call_agent)
workflow.set_entry_point("agent")
web_agent = workflow.compile()