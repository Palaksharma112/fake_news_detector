from rapidfuzz import fuzz

TRUSTED_DOMAINS = {
    "reuters.com": 30,
    "apnews.com": 30,
    "bbc.com": 25,
    "thehindu.com": 20,
    "indianexpress.com": 20,
    "hindustantimes.com": 20,
    "ndtv.com": 15,
    "indiatoday.in": 15,
    "timesofindia.indiatimes.com": 15,
    "livemint.com": 15,
    "business-standard.com": 15,
    "economictimes.indiatimes.com": 15
}


def verify_sources(results, claim, image_score=None):
    """
    Verify claim using trusted web sources.
    Optionally combine with image analysis score.
    """

    claim = claim.lower().strip()

    best_similarity = 0
    credibility_score = 0
    trusted_links = []

    for r in results:

        title = r.get("title", "").lower()
        snippet = r.get("snippet", "").lower()

        combined = title + " " + snippet

        similarity = fuzz.token_set_ratio(claim, combined)

        best_similarity = max(best_similarity, similarity)

        link = r.get("link", "").lower()

        for domain, weight in TRUSTED_DOMAINS.items():
            if domain in link:
                credibility_score += weight
                trusted_links.append(r)
                break

    credibility_score = min(credibility_score, 100)

    web_score = (
        best_similarity * 0.6 +
        credibility_score * 0.4
    )

    # Combine image analysis if available
    if image_score is not None:
        final_score = (
            web_score * 0.7 +
            image_score * 0.3
        )
    else:
        final_score = web_score

    if final_score >= 80:
        verdict = "REAL"

    elif final_score >= 65:
        verdict = "LIKELY REAL"

    elif final_score >= 45:
        verdict = "UNCERTAIN"

    else:
        verdict = "LIKELY FAKE"

    return {
        "verdict": verdict,
        "similarity": round(best_similarity, 2),
        "credibility": round(credibility_score, 2),
        "web_score": round(web_score, 2),
        "final_score": round(final_score, 2),
        "sources": trusted_links
    }