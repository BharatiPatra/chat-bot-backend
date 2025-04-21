# pip install -q -U google-genai to use gemini as a client

import os
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv
from utils.constants import MODEL_ID

from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.ollama import ollama_embed


import asyncio
import nest_asyncio

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")
WORKING_DIR = "./knowledge_base"

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


async def initialize_rag():
    rag = LightRAG(
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

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

# Initialize RAG instance
rag = asyncio.run(initialize_rag())
# Insert text into the RAG instance
# def insert_text(rag, file_path):

#     if not(os.path.exists(WORKING_DIR)):
#         os.mkdir(WORKING_DIR)

#     file_path = "/Users/kishorkumar/Python_projects/quant4trading/AshaAI/v2/book.txt"
#     with open(file_path, "r") as file:
#         text = file.read()

#     rag.insert(text)
import os

def insert_text(folder_path):
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                rag.insert(text)

