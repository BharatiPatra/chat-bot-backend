from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal
from agents.core.agent import core_agent

app = FastAPI()

# Pydantic model for user input/output
class Chat(BaseModel):
    id: Literal["human", "ai"]
    query: str

# GET endpoint
@app.get("/api/test")
def test():
    return "testing"

# POST endpoint
@app.post("/api/users", response_model=List[Chat])
def get_query_result(chats: List[Chat]):
    chat_history = [
        {"role": "human", "content": "Hi there"},
        {"role": "ai", "content": "Hello! How can I help you?"},
        {"role": "human", "content": "Tell me about available jobs"},
        {"role": "ai", "content": "We have job openings in software and data science."}
    ]
    new_query = "Can you check java job openings?"

    inputs = {
        "user_query": new_query,
        "chat_history": chat_history,
        "tool_results": {},  # optional but safe to include
        "intent": "",        # will be set by extract_intent
        "response": "",      # will be set by generate_response
    }

    result = core_agent.invoke(inputs)
    return {"role": "ai", "content": result["response"]}
