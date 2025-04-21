from bs4 import BeautifulSoup

def parse(url, html):
    text = BeautifulSoup(html, "html.parser").get_text(strip=True)
    # soup = BeautifulSoup(html, "html.parser")
    # jobs = []
    # for job_card in soup.find_all("div"):
    #     text = job_card.get_text(strip=True)
    #     jobs.append({
    #         "apply_link": url,
    #         "raw": text
    #     })
    return text
