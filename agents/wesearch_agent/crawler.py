import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from agents.wesearch_agent.extractor import extract_jobs_from_html

async def crawl_urls(urls):
    browser_config = BrowserConfig(headless=True)
    crawl_config = CrawlerRunConfig()
    results = []

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for url in urls:
            try:
                result = await crawler.arun(url=url, config=crawl_config)
                text = extract_jobs_from_html(url, result.html)
                results.append({"link":url, "text":text})
                # results.extend(job_data)
            except Exception as e:
                print(f"Error crawling {url}: {e}")
    return results
