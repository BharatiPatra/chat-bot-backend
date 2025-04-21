from agents.wesearch_agent.searcher import search_duckduckgo
from agents.wesearch_agent.crawler import crawl_urls
import asyncio

async def run_wesearch(query):
    urls = search_duckduckgo(query, max_results=10)
    jobs = await crawl_urls(urls)
    return jobs

if __name__ == "__main__":
    query = "Java 5 year experience job"
    jobs = run_wesearch(query)
    for job in jobs:
        print(job)
