from agents.feature.light_rag import rag
from lightrag import QueryParam

def feature_agent(user_query: str):
    """
    Function to query the FAQ agent.
    Args:
        user_query (str): The query string.
    Returns:
        str: The response from the FAQ agent.
    """
    response = rag.query(
        query=user_query,
        param=QueryParam(mode="mix", top_k=5),
    )
    return response