import requests
from bs4 import BeautifulSoup
import urllib.parse
import re

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
    )
}

STOP_WORDS = {
    "the", "a", "an", "is", "was", "were", "has", "have",
    "had", "into", "of", "for", "to", "and", "or",
    "in", "on", "with", "from", "this", "that"
}


def extract_keywords(text):
    words = re.findall(r"[A-Za-z0-9]+", text)

    keywords = [
        w for w in words
        if len(w) > 2 and w.lower() not in STOP_WORDS
    ]

    return " ".join(keywords[:10])


def search_once(query):

    url = (
        "https://duckduckgo.com/html/?q=" +
        urllib.parse.quote(query)
    )

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=10
        )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        results = []

        for item in soup.select(".result"):

            title = item.select_one(".result__a")
            snippet = item.select_one(".result__snippet")

            if title is None:
                continue

            results.append({
                "title": title.get_text(" ", strip=True),
                "link": title.get("href"),
                "snippet": (
                    snippet.get_text(" ", strip=True)
                    if snippet else ""
                )
            })

        return results

    except Exception as e:

        print("Search Error:", e)
        return []


def search_web(query):
    """
    Works for:
    1. News text
    2. OCR text
    3. AI-generated image captions
    """

    if not query.strip():
        return []

    searches = [
        query,
        extract_keywords(query),
        query + " news",
        query + " Reuters",
        query + " BBC",
        query + " AP News"
    ]

    results = []
    seen = set()

    for q in searches:

        if not q.strip():
            continue

        for item in search_once(q):

            link = item["link"]

            if link in seen:
                continue

            seen.add(link)
            results.append(item)

    return results[:25]