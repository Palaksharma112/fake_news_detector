import requests
from bs4 import BeautifulSoup
import urllib.parse
import re

def extract_keywords(text):

    words = re.findall(r"[A-Za-z]+", text)

    stop_words = {
        "the","a","an","is","was","were",
        "has","have","had","into","of",
        "for","to","and","or","in","on"
    }

    keywords = [
        w for w in words
        if len(w) > 3 and w.lower() not in stop_words
    ]

    return " ".join(keywords[:8])


def search_once(query):

    url = f"https://duckduckgo.com/html/?q={urllib.parse.quote(query)}"

    headers = {
        "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        results = []

        for result in soup.select(".result"):

            title_tag = result.select_one(".result__a")
            snippet_tag = result.select_one(".result__snippet")

            if not title_tag:
                continue

            results.append({
                "title": title_tag.get_text(" ", strip=True),
                "link": title_tag.get("href"),
                "snippet": (
                    snippet_tag.get_text(" ", strip=True)
                    if snippet_tag else ""
                )
            })

        return results

    except Exception as e:

        print("Search Error:", e)
        return []


def search_web(claim):

    searches = [
        claim,
        extract_keywords(claim),
        claim + " news",
        claim + " Reuters",
        claim + " BBC"
    ]

    final_results = []
    seen = set()

    for q in searches:

        for item in search_once(q):

            link = item["link"]

            if link not in seen:

                seen.add(link)
                final_results.append(item)

    return final_results[:20]