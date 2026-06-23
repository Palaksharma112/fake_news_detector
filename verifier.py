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

def verify_sources(results, claim):

    claim = claim.lower()

    best_similarity = 0
    credibility_score = 0

    trusted_links = []

    for r in results:

        title = r.get("title", "").lower()
        snippet = r.get("snippet", "").lower()

        text = title + " " + snippet

        similarity = fuzz.token_set_ratio(
            claim,
            text
        )

        best_similarity = max(
            best_similarity,
            similarity
        )

        link = r.get("link", "").lower()

        for domain, weight in TRUSTED_DOMAINS.items():

            if domain in link:

                credibility_score += weight
                trusted_links.append(r)
                break

    credibility_score = min(
        credibility_score,
        100
    )

    final_score = (
        best_similarity * 0.6 +
        credibility_score * 0.4
    )

    if final_score >= 75:
        verdict = "REAL"

    elif final_score >= 55:
        verdict = "LIKELY REAL"

    elif final_score >= 40:
        verdict = "UNCERTAIN"

    else:
        verdict = "UNVERIFIED"

    return {
        "verdict": verdict,
        "similarity": round(best_similarity, 2),
        "credibility": round(credibility_score, 2),
        "final_score": round(final_score, 2),
        "sources": trusted_links
    }