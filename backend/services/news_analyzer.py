"""
News Analyzer — API-first, no ML model required.
Uses Google Fact Check, ClaimBuster, NewsAPI + heuristics to produce
a clean verdict with result statement and related links.
"""

import re
import logging
import yaml

logger = logging.getLogger(__name__)


def _load_api_keys(config_path="configs/system_configs.yaml") -> dict:
    try:
        with open(config_path) as f:
            return yaml.safe_load(f).get("news_apis", {})
    except Exception:
        return {}


def _build_result_statement(prediction: str, confidence: float, signals: list,
                             fact_check_results: list, cross_ref: int) -> str:
    """Generate a human-readable single result statement."""
    pct = round(confidence * 100)

    if fact_check_results:
        false_checks = [r for r in fact_check_results if r.get("verdict") == "false"]
        if false_checks:
            publisher = false_checks[0].get("publisher", "fact-checkers")
            rating    = false_checks[0].get("rating", "false")
            return (f"This claim has been fact-checked and rated \"{rating}\" by {publisher}. "
                    f"Our analysis is {pct}% confident this is misinformation.")

        true_checks = [r for r in fact_check_results if r.get("verdict") == "true"]
        if true_checks:
            publisher = true_checks[0].get("publisher", "fact-checkers")
            return (f"This claim has been verified as accurate by {publisher}. "
                    f"Analysis is {pct}% confident this is legitimate news.")

    # Check if gravity flags triggered
    gravity_signals = [s for s in signals if "living public figure" in s or "Extraordinary" in s]
    if gravity_signals and prediction == "fake":
        return (f"This claim makes an extraordinary assertion with no source attribution. "
                f"Death hoaxes and unverified claims about public figures are among the most "
                f"common forms of misinformation. Analysis is {pct}% confident this is false.")

    if prediction == "fake":
        if cross_ref == 0:
            return (f"No credible news sources found covering this story. "
                    f"Combined analysis rates this {pct}% likely to be misinformation.")
        return (f"Analysis detected suspicious patterns in this content. "
                f"Rated {pct}% likely to be misleading or false.")
    else:
        if cross_ref >= 3:
            return (f"This story is covered by {cross_ref} credible news sources. "
                    f"Analysis is {pct}% confident this is legitimate news.")
        return (f"No strong indicators of misinformation detected. "
                f"Analysis is {pct}% confident this content appears legitimate.")


def _heuristic_score(text: str) -> dict:
    """Fast heuristic scoring — no model needed."""
    text_lower = text.lower()

    FAKE_PATTERNS = [
        r"\byou won't believe\b", r"\bshocking\b", r"\bbreaking\b",
        r"\bexplosive\b", r"\bsecret\b", r"\bthey don't want you to know\b",
        r"\bmiracle\b", r"\bhoax\b", r"\bconspiracy\b", r"\bscam\b",
        r"\bfake news\b", r"\blie\b", r"\bfraud\b", r"\bviral\b",
    ]
    CREDIBLE_PATTERNS = [
        r"\baccording to\b", r"\bresearch shows\b", r"\bstudies indicate\b",
        r"\bexperts say\b", r"\bofficial statement\b", r"\bconfirmed by\b",
        r"\breports suggest\b", r"\bsources say\b",
    ]

    fake_hits     = sum(1 for p in FAKE_PATTERNS if re.search(p, text_lower))
    credible_hits = sum(1 for p in CREDIBLE_PATTERNS if re.search(p, text_lower))
    exclaim_ratio = text.count("!") / max(len(text), 1)
    caps_ratio    = sum(1 for c in text if c.isupper()) / max(len(text), 1)

    score = 0.3
    score += min(fake_hits * 0.07, 0.28)
    score -= min(credible_hits * 0.06, 0.18)
    score += min(exclaim_ratio * 5, 0.10)
    score += min(caps_ratio * 2, 0.10)

    # Gravity check overrides base score for extraordinary claims
    gravity = _gravity_check(text)
    if gravity["boost"] > score:
        score = gravity["boost"]

    fake_prob = max(0.05, min(0.95, score))

    anomalies = list(gravity["flags"])
    if fake_hits > 2:
        anomalies.append(f"{fake_hits} clickbait/sensational phrases detected")
    if caps_ratio > 0.1:
        anomalies.append("excessive capitalization")
    if exclaim_ratio > 0.02:
        anomalies.append("excessive exclamation marks")
    if credible_hits == 0 and fake_prob > 0.5:
        anomalies.append("no credibility indicators found")

    return {
        "fake_probability": round(fake_prob, 4),
        "anomalies": anomalies,
        "word_count": len(text.split()),
        "fake_pattern_hits": fake_hits,
        "credible_pattern_hits": credible_hits,
    }


# Extraordinary claim patterns — these need cross-reference confirmation
_EXTRAORDINARY_PATTERNS = [
    r"\b(is|was|has been|found|declared|confirmed|reported)\s+(dead|killed|assassinated|murdered)\b",
    r"\b(died|passed away|no more|no longer alive)\b",
    r"\b(arrested|jailed|imprisoned)\s+(the\s+)?(president|prime minister|pm|ceo|king|queen)\b",
    r"\b(world war|nuclear\s+(attack|bomb|war|strike))\b",
    r"\b(cure|cures|cured)\s+(cancer|hiv|aids|covid|diabetes)\b",
    r"\b(aliens?|ufos?)\s+(land|landed|attack|confirmed|real|exist)\b",
    r"\bgovernment\s+(hiding|hid|covered up|suppressing)\b",
    r"\b(vaccine|vaccines)\s+(kill|kills|killed|cause|causes|caused)\b",
    r"\bearth\s+is\s+flat\b",
]

# Named figures whose death claims are almost always hoaxes (actively living as of 2025)
# NOTE: Do NOT add people who have died — this list is for living people only
_CONFIRMED_LIVING = [
    "narendra modi", "pm modi",
    "donald trump",
    "vladimir putin",
    "xi jinping",
    "elon musk",
    "bill gates",
    "pope francis",
    "volodymyr zelensky",
    "emmanuel macron",
    "virat kohli",
    "mark zuckerberg",
    "jeff bezos",
    "taylor swift",
    "amitabh bachchan",
    "shah rukh khan",
]


def _gravity_check(text: str) -> dict:
    """
    Detect extraordinary claims that need cross-reference confirmation.
    Returns a SOFT suspicion boost — NOT a hard override.
    The aggregate_verdict will cancel this if credible sources confirm the claim.
    """
    text_lower = text.lower()
    boost = 0.0
    flags = []
    is_death_claim = False

    death_re = re.compile(r"\b(dead|died|killed|assassinated|murdered|passed away|no more|death)\b")

    # Only flag death claims about people we are CERTAIN are alive right now
    if death_re.search(text_lower):
        for figure in _CONFIRMED_LIVING:
            if figure in text_lower:
                boost = max(boost, 0.70)
                flags.append(f"Unverified death claim about confirmed living person: '{figure}'")
                is_death_claim = True
                break

    # Other extraordinary patterns — softer signal
    if not is_death_claim:
        for pattern in _EXTRAORDINARY_PATTERNS:
            if re.search(pattern, text_lower):
                boost = max(boost, 0.55)
                flags.append("Extraordinary claim — requires cross-reference verification")
                break

    # Short + no source = more suspicious, but still soft
    has_source = bool(re.search(
        r"\b(according to|reported by|confirmed by|sources say|official|statement)\b",
        text_lower
    ))
    if boost > 0 and len(text.split()) < 20 and not has_source:
        boost = min(boost + 0.08, 0.90)
        flags.append("No source attribution for a major claim")

    return {"boost": boost, "flags": flags, "is_death_claim": is_death_claim}


def analyze_news(text_or_url: str, config_path="configs/system_configs.yaml") -> dict:
    """
    Full news analysis — API fact-checking + heuristics.
    No ML model required.
    Returns clean structured result with verdict, statement, and links.
    """
    from backend.services.news_fact_checker import (
        check_google_factcheck, check_claimbuster,
        search_newsapi, search_related_links, aggregate_verdict,
        check_groq, check_gemini, get_smart_search_query, get_groq_related_links,
    )

    # Resolve URL to text if needed
    source_url = None
    if text_or_url.startswith("http://") or text_or_url.startswith("https://"):
        source_url = text_or_url
        text = _fetch_url_text(text_or_url)
        if not text:
            return {"error": "Failed to fetch URL content", "content_type": "news"}
    else:
        text = text_or_url

    if len(text.strip()) < 10:
        return {"error": "Text too short for analysis", "content_type": "news"}

    # Heuristic analysis
    heuristic = _heuristic_score(text)

    # API keys
    keys = _load_api_keys(config_path)

    # Run all available APIs
    claim = text[:500]  # use first 500 chars as the claim
    google_fc   = check_google_factcheck(claim, keys.get("google_factcheck_api_key", ""))
    claimbuster = check_claimbuster(claim, keys.get("claimbuster_api_key", ""))
    groq_result = check_groq(claim, keys.get("groq_api_key", ""))
    gemini_result = check_gemini(claim, keys.get("gemini_api_key", ""))

    # Use Groq to generate a smarter search query for better related links
    smart_query = get_smart_search_query(claim, keys.get("groq_api_key", ""))
    search_claim = smart_query if smart_query and smart_query != claim else claim.strip().lower().capitalize()

    newsapi = search_newsapi(search_claim, keys.get("newsapi_key", ""))
    related = search_related_links(
        search_claim,
        keys.get("google_custom_search_api_key", ""),
        keys.get("google_custom_search_cx", ""),
    )

    # Aggregate verdict
    heuristic_input = {"fake_probability": heuristic["fake_probability"]}
    verdict = aggregate_verdict(heuristic_input, google_fc, claimbuster, newsapi, related,
                                gemini=gemini_result, groq=groq_result, claim_text=claim)

    # Inject Groq AI verdict as a synthetic related link if available
    if groq_result.get("available") and groq_result.get("explanation"):
        ai_source = groq_result.get("source", "AI Fact-Check")
        ai_verdict_label = groq_result.get("verdict", "").upper()
        # Only inject if there's a meaningful explanation, skip the groq.com placeholder
        pass  # AI verdict is shown separately in the UI, not as a link

    # Use Groq to find real relevant fact-check links
    groq_links = get_groq_related_links(claim, keys.get("groq_api_key", ""))
    for link in groq_links:
        verdict["related_links"].append(link)

    # Build human-readable statement
    statement = _build_result_statement(
        verdict["prediction"],
        verdict["confidence_score"],
        verdict["signals"],
        verdict["fact_check_results"],
        verdict["cross_reference_count"],
    )

    # APIs used
    apis_used = []
    if google_fc.get("available"):      apis_used.append("Google Fact Check")
    if claimbuster.get("available"):    apis_used.append("ClaimBuster")
    if newsapi.get("available"):        apis_used.append("NewsAPI")
    if related.get("available"):        apis_used.append("Google Custom Search")
    if groq_result.get("available"):    apis_used.append("Groq/Llama-3.3-70b")
    elif gemini_result.get("available"):apis_used.append("Gemini-2.0-flash")
    if not apis_used:                   apis_used.append("Heuristic Analysis")

    # Pick the active AI result for display
    ai_verdict_display = None
    if groq_result.get("available"):
        ai_verdict_display = groq_result
    elif gemini_result.get("available"):
        ai_verdict_display = gemini_result

    return {
        "content_type": "news",
        "prediction":        verdict["prediction"],
        "confidence_score":  verdict["confidence_score"],
        "fake_probability":  verdict["fake_probability"],
        "result_statement":  statement,
        "signals":           verdict["signals"],
        "related_links":     verdict["related_links"],
        "fact_check_results": verdict["fact_check_results"],
        "cross_reference_count": verdict["cross_reference_count"],
        "claimbuster_score": verdict["claimbuster_score"],
        "anomalies":         heuristic["anomalies"],
        "word_count":        heuristic["word_count"],
        "source_url":        source_url,
        "text_preview":      text[:300] + "..." if len(text) > 300 else text,
        "apis_used":         apis_used,
        "model_used":        "API Fact-Check + Heuristic Analyzer",
        "ai_verdict":        ai_verdict_display,
        # Mapped fields for report_generator
        "text_features": {
            "clickbait_indicators":   heuristic["fake_pattern_hits"],
            "credibility_indicators": heuristic["credible_pattern_hits"],
        },
        "fact_check_apis": {
            "google_factcheck_available": google_fc.get("available", False),
            "claimbuster_available":      claimbuster.get("available", False),
            "newsapi_available":          newsapi.get("available", False),
        },
        "possible_generation_method": (
            ai_verdict_display.get("explanation", "AI-generated or human-written misinformation")
            if ai_verdict_display and ai_verdict_display.get("verdict") == "fake"
            else "AI-generated or human-written misinformation"
        ),
        "detected_anomalies": (
            ([ai_verdict_display.get("explanation")] if ai_verdict_display and ai_verdict_display.get("verdict") == "fake" and ai_verdict_display.get("explanation") else []) +
            (ai_verdict_display.get("red_flags", []) if ai_verdict_display else []) +
            heuristic["anomalies"]
        ),
    }


def _fetch_url_text(url: str) -> str:
    """Fetch and extract plain text from a URL."""
    try:
        import urllib.request
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.parts = []
                self._skip = False
                self._skip_tags = {"script", "style", "nav", "footer", "header"}

            def handle_starttag(self, tag, attrs):
                if tag in self._skip_tags:
                    self._skip = True

            def handle_endtag(self, tag):
                if tag in self._skip_tags:
                    self._skip = False

            def handle_data(self, data):
                if not self._skip and len(data.strip()) > 20:
                    self.parts.append(data.strip())

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        parser = TextExtractor()
        parser.feed(html)
        return " ".join(parser.parts[:60])
    except Exception:
        return ""
