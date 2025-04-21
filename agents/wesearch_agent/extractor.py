from agents.wesearch_agent.parsers import default_parser

def extract_jobs_from_html(url, html):
    return default_parser.parse(url, html)
