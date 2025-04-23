# pip install -q -U google-genai to use gemini as a client
import sys
from typing import Optional
sys.path.append('./libraries/PathRAG')

import os
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv
from utils.constants import MODEL_ID

from libraries.PathRAG.PathRAG.utils import EmbeddingFunc
from libraries.PathRAG.PathRAG import PathRAG, QueryParam
from libraries.PathRAG.PathRAG.llm import ollama_embed
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI




import nest_asyncio

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")
WORKING_DIR = "./knowledge_base/path_rag"

llm = ChatGoogleGenerativeAI(model=MODEL_ID, temperature=0.0)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    # 1. Initialize the GenAI Client with your Gemini API Key
    client = genai.Client(api_key=gemini_api_key)

    # 2. Combine prompts: system prompt, history, and user prompt
    if history_messages is None:
        history_messages = []

    combined_prompt = ""
    if system_prompt:
        combined_prompt += f"{system_prompt}\n"

    for msg in history_messages:
        # Each msg is expected to be a dict: {"role": "...", "content": "..."}
        combined_prompt += f"{msg['role']}: {msg['content']}\n"

    # Finally, add the new user prompt
    combined_prompt += f"user: {prompt}"

    # 3. Call the Gemini model
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[combined_prompt],
        config=types.GenerateContentConfig(max_output_tokens=500, temperature=0.1),
    )

    # 4. Return the response text
    return response.text


def initialize_rag():
    rag = PathRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model="nomic-embed-text", host="http://localhost:11434"
            ),
        ),
    )

    return rag

# Initialize RAG instance
rag = initialize_rag()

def insert_text(folder_path):
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                rag.insert(text)

class AgentState(BaseModel):
    history: list
    compiled_history: Optional[str]
    query: str
    previous_response: Optional[str]
    response: Optional[str]


def run_history_summary(state: AgentState):
    history = state.history

    # If history is a list and empty, skip LLM call
    if isinstance(history, list) and len(history) == 0:
        state.compiled_history = ""
        return state

    # If history is already a summarized string
    if isinstance(history, str) and not history.strip():
        state.compiled_history = ""
        return state
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a summarization agent tasked with reviewing a conversation between a human user and an AI assistant. Your goal is to extract and concisely present the core topics discussed, key insights shared by the assistant, and any decisions, requests, or conclusions made by the user.

                Instructions:

                Summarize the core discussion – Identify the main topics and goals of the conversation.

                Highlight key information – Extract important facts, advice, or insights given by the assistant.

                Note user actions or intentions – Capture any requests, goals, or plans mentioned by the user.

                Keep it concise and clear – Use bullet points or short paragraphs as needed.
                """,
            ),
            ("user", "{chat_history}"),
        ]
    )
    history = state.history

    formatted_history = ""
    if isinstance(history, list):
        for msg in history:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            formatted_history += f"{role}: {content}\n"
    else:
        formatted_history = history  # if already a string (e.g., summarized)

    prompt = prompt.format_prompt(chat_history=formatted_history)
    response = llm.invoke(prompt)
    state.compiled_history = response.content  
    return state

def run_query_rag(state: AgentState):
    query = state.query
    compiled_history = state.compiled_history
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("user", "Conversation Summary: {compiled_history}"),
            ("user", "{query}"),
        ]
    )
    prompt = prompt.format_prompt(compiled_history=compiled_history, query=query).to_string()
    response = rag.query(prompt, param=QueryParam(mode="hybrid"))
    state.previous_response = response
    return state

def run_response_summarize(state: AgentState):
    query = state.query
    previous_response = state.previous_response
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system", 
                """
                You have received two pieces of information:
                1. **User Query**: The original query from the user.
                2. **Previous Agent Response**: The response generated by the first agent based on the query.

                Your task is to evaluate whether the response is relevant to the user's query.
                - If the response is relevant, return it as is.
                - If the response is not relevant, respond with: "This query is not part of this project." 
                - Alternatively, you can suggest contacting support by saying: "It looks like this query might be better handled by our customer support team. You can reach them at https://herkey.com/support."
                """
            ),
            ("user", "User Query:\n{query}"),
            ("ai", "Previous Agent Response:\n{previous_response}"),
        ]
    )
    prompt = prompt.format_prompt(query=query, previous_response=previous_response)
    response = llm.invoke(prompt)
    state.response = response.content
    return state