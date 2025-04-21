from duckduckgo_search import DDGS

def search_duckduckgo(query, max_results=10):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, region='india', safesearch='Off', max_results=max_results):
            results.append(r["href"])
    return results
