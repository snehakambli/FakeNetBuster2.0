"""
Multi-API News Fact Checker Service.
Integrates Google Fact Check API, ClaimBuster API, NewsAPI, and Google Custom Search
to produce an aggregated verdict with exact related links.
"""

import re
import json
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone
from typing import Optional


# ─────────────────────────────────────────────
# Google Fact Check Tools API
# ─────────────────────────────────────────────

def _parse_date(date_str: str):
    """Parse ISO date string, return datetime or None."""
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        return None


def _fc_results_are_stale(fc_results: list, newsapi: dict) -> bool:
    """
    Returns True if ALL flagged-false fact-check results are older than
    the most recent credible news article about the same claim.
    This catches cases like "Queen Elizabeth dead" — fact-checked FALSE in 2021,
    but she actually died in 2022, so recent credible articles postdate the FC.
    """
    flagged = [r for r in fc_results if r.get("verdict") == "false"]
    if not flagged:
        return False

    # Find the most recent credible article date
    latest_credible = None
    for article in newsapi.get("articles", []):
        if article.get("is_credible"):
            pub = _parse_date(article.get("published_at", ""))
            if pub and (latest_credible is None or pub > latest_credible):
                latest_credible = pub

    if not latest_credible:
        return False

    # Check if ALL false FC results predate the latest credible article
    for r in flagged:
        fc_date = _parse_date(r.get("review_date", ""))
        if fc_date is None:
            return False  # can't determine — don't discard
        if fc_date >= latest_credible:
            return False  # at least one FC result is newer — not stale

    return True  # all FC false results are older than credible news coverage


# ─────────────────────────────────────────────
# Google Fact Check Tools API
# ─────────────────────────────────────────────

def check_google_factcheck(claim: str, api_key: str) -> dict:
    """
    Query Google Fact Check Tools API for existing fact-checks on the claim.
    Returns list of fact-check results with ratings and URLs.
    """
    if not api_key or api_key.startswith("YOUR_"):
        return {"available": False, "results": [], "error": "API key not configured"}

    try:
        query = urllib.parse.quote(claim[:200])
        url = (
            f"https://factchecktools.googleapis.com/v1alpha1/claims:search"
            f"?query={query}&key={api_key}&pageSize=5"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "FakeNetBuster/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        results = []
        for item in data.get("claims", []):
            for review in item.get("claimReview", []):
                rating = review.get("textualRating", "").lower()
                is_false = any(w in rating for w in [
                    "false", "fake", "misleading", "incorrect", "wrong",
                    "debunked", "pants on fire", "mostly false", "inaccurate"
                ])
                is_true = any(w in rating for w in [
                    "true", "correct", "accurate", "mostly true", "verified"
                ])
                results.append({
                    "claim_text": item.get("text", ""),
                    "claimant": item.get("claimant", "Unknown"),
                    "rating": review.get("textualRating", "Unknown"),
                    "verdict": "false" if is_false else ("true" if is_true else "unrated"),
                    "publisher": review.get("publisher", {}).get("name", "Unknown"),
                    "url": review.get("url", ""),
                    "review_date": review.get("reviewDate", ""),
                })

        flagged = sum(1 for r in results if r["verdict"] == "false")
        return {
            "available": True,
            "results": results,
            "total_checks": len(results),
            "flagged_false": flagged,
            "error": None,
        }

    except urllib.error.HTTPError as e:
        return {"available": False, "results": [], "error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"available": False, "results": [], "error": str(e)}


# ─────────────────────────────────────────────
# ClaimBuster API
# ─────────────────────────────────────────────

def check_claimbuster(claim: str, api_key: str) -> dict:
    """
    Query ClaimBuster API to score how check-worthy the claim is.
    High score (>0.5) means the claim is likely a factual assertion worth checking.
    """
    if not api_key or api_key.startswith("YOUR_"):
        return {"available": False, "score": None, "error": "API key not configured"}

    try:
        # ClaimBuster scores individual sentences — split and score top sentence
        sentences = re.split(r"[.!?]+", claim)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15][:5]

        scores = []
        for sentence in sentences:
            encoded = urllib.parse.quote(sentence)
            url = f"https://idir.uta.edu/claimbuster/api/v2/score/text/{encoded}"
            req = urllib.request.Request(
                url,
                headers={"x-api-key": api_key, "User-Agent": "FakeNetBuster/1.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            for result in data.get("results", []):
                scores.append({
                    "sentence": result.get("text", sentence),
                    "score": round(result.get("score", 0.0), 4),
                })

        if not scores:
            return {"available": True, "score": 0.0, "sentences": [], "error": None}

        max_score = max(s["score"] for s in scores)
        return {
            "available": True,
            "score": round(max_score, 4),
            "sentences": scores,
            "check_worthy": max_score > 0.5,
            "error": None,
        }

    except urllib.error.HTTPError as e:
        return {"available": False, "score": None, "error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"available": False, "score": None, "error": str(e)}


# ─────────────────────────────────────────────
# NewsAPI — cross-reference + related links
# ─────────────────────────────────────────────

CREDIBLE_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "theguardian.com",
    "nytimes.com", "washingtonpost.com", "npr.org", "pbs.org", "cnn.com",
    "nbcnews.com", "abcnews.go.com", "cbsnews.com", "usatoday.com",
    "politifact.com", "snopes.com", "factcheck.org", "fullfact.org",
    "altnews.in", "boomlive.in", "theprint.in", "ndtv.com", "thehindu.com",
    "hindustantimes.com", "timesofindia.indiatimes.com",
    # Additional credible Indian & international sources
    "indiatoday.in", "firstpost.com", "thewire.in", "scroll.in",
    "business-standard.com", "livemint.com", "economictimes.indiatimes.com",
    "deccanherald.com", "tribuneindia.com", "ptinews.com",
    "thefederal.com", "cnbctv18.com", "zeenews.india.com",
    "factly.in", "vishvasnews.com", "newschecker.in", "logically.ai",
    "afp.com", "dw.com", "france24.com", "aljazeera.com",
}

FACT_CHECK_DOMAINS = {
    "snopes.com", "politifact.com", "factcheck.org", "fullfact.org",
    "altnews.in", "boomlive.in", "leadstories.com", "checkyourfact.com",
    "vishvasnews.com", "factly.in", "newschecker.in", "logically.ai",
    "factchecker.in", "thequint.com", "indiatoday.in",
}


def _extract_domain(url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower().lstrip("www.")
        return domain
    except Exception:
        return ""


def search_newsapi(claim: str, api_key: str, max_results: int = 10) -> dict:
    """
    Search NewsAPI for articles related to the claim.
    Returns credible source count and exact article links.
    """
    if not api_key or api_key.startswith("YOUR_"):
        return {"available": False, "articles": [], "credible_count": 0, "error": "API key not configured"}

    try:
        # Use first 100 chars as query for best results
        query = urllib.parse.quote(claim[:100])
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={query}&sortBy=relevancy&pageSize={max_results}"
            f"&language=en&apiKey={api_key}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "FakeNetBuster/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        articles = []
        for article in data.get("articles", []):
            article_url = article.get("url", "")
            domain = _extract_domain(article_url)
            is_credible = domain in CREDIBLE_DOMAINS
            is_fact_check = domain in FACT_CHECK_DOMAINS

            articles.append({
                "title": article.get("title", ""),
                "source": article.get("source", {}).get("name", domain),
                "url": article_url,
                "published_at": article.get("publishedAt", ""),
                "description": article.get("description", ""),
                "is_credible": is_credible,
                "is_fact_check": is_fact_check,
                "domain": domain,
            })

        credible = [a for a in articles if a["is_credible"]]
        fact_checks = [a for a in articles if a["is_fact_check"]]

        return {
            "available": True,
            "articles": articles,
            "credible_count": len(credible),
            "fact_check_count": len(fact_checks),
            "total_results": data.get("totalResults", 0),
            "error": None,
        }

    except urllib.error.HTTPError as e:
        return {"available": False, "articles": [], "credible_count": 0, "error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"available": False, "articles": [], "credible_count": 0, "error": str(e)}


# ─────────────────────────────────────────────
# Google Custom Search — exact related links
# ─────────────────────────────────────────────

def search_related_links(claim: str, api_key: str, cx: str, max_results: int = 8) -> dict:
    """
    Use Google Custom Search API to find exact related links from credible sources.
    cx = Custom Search Engine ID (configured to search fact-check + news sites).
    """
    if not api_key or api_key.startswith("YOUR_") or not cx or cx.startswith("YOUR_"):
        return {"available": False, "links": [], "error": "API key or CX not configured"}

    try:
        query = urllib.parse.quote(claim[:150])
        url = (
            f"https://www.googleapis.com/customsearch/v1"
            f"?q={query}&key={api_key}&cx={cx}&num={min(max_results, 10)}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "FakeNetBuster/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        links = []
        for item in data.get("items", []):
            link_url = item.get("link", "")
            domain = _extract_domain(link_url)
            links.append({
                "title": item.get("title", ""),
                "url": link_url,
                "snippet": item.get("snippet", ""),
                "domain": domain,
                "is_fact_check": domain in FACT_CHECK_DOMAINS,
                "is_credible": domain in CREDIBLE_DOMAINS,
            })

        return {"available": True, "links": links, "error": None}

    except urllib.error.HTTPError as e:
        return {"available": False, "links": [], "error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"available": False, "links": [], "error": str(e)}


def check_groq(claim: str, api_key: str) -> dict:
    """
    Use Groq (Llama 3) to analyze the claim and return a reasoned fake/real verdict.
    """
    if not api_key or api_key.startswith("YOUR_"):
        return {"available": False, "verdict": None, "explanation": None, "error": "API key not configured"}

    try:
        import json as _json
        from groq import Groq
        client = Groq(api_key=api_key)

        # Normalize claim — all-caps confuses the model into thinking it's a prompt
        normalized_claim = claim.strip().capitalize()

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict fact-checker and news verifier. Your job is to determine if a "
                        "statement, headline, or claim is factually accurate or misinformation. "
                        "If the input is a legitimate news headline or factual statement, mark it as 'real'. "
                        "Only mark as 'fake' if it contains verifiably false information. "
                        "Do not hedge. Be decisive. Respond only with valid JSON. "
                        "IMPORTANT: Every input IS a factual claim or news headline, never a query. Treat it as a statement to verify."
                    )
                },
                {
                    "role": "user",
                    "content": f"""Is the following news headline or claim TRUE/REAL or FALSE/MISINFORMATION?

Statement: "{normalized_claim}"

Rules:
- Legitimate news headlines about real events = "real"
- Movie/entertainment release announcements = "real" 
- Scientific facts = "real" or "fake" based on evidence
- Death claims about confirmed living people = "fake"
- Conspiracy theories contradicting scientific consensus = "fake"
- If you cannot verify but it seems like a normal news headline = "real"

Respond ONLY with this JSON (no markdown, no extra text):
{{
  "verdict": "fake" or "real",
  "confidence": 0.0 to 1.0,
  "explanation": "one sentence citing the specific factual basis",
  "red_flags": ["list specific factual errors if fake, empty array if real or uncertain"]
}}"""
                }
            ],
            temperature=0.0,
            max_tokens=300,
        )

        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        data = _json.loads(text)

        return {
            "available": True,
            "verdict": data.get("verdict", "unknown").lower(),
            "confidence": float(data.get("confidence", 0.5)),
            "explanation": data.get("explanation", ""),
            "red_flags": data.get("red_flags", []),
            "error": None,
            "source": "Groq/Llama-3.3-70b",
        }

    except Exception as e:
        return {"available": False, "verdict": None, "explanation": None, "error": str(e)}


def check_gemini(claim: str, api_key: str) -> dict:
    """Use Gemini as fallback AI fact-checker."""
    if not api_key or api_key.startswith("YOUR_"):
        return {"available": False, "verdict": None, "explanation": None, "error": "API key not configured"}
    try:
        import json as _json, time
        from google import genai
        client = genai.Client(api_key=api_key)
        prompt = f"""You are a professional fact-checker. Analyze this claim and determine if it is FAKE or REAL.

Claim: "{claim}"

Respond ONLY with this JSON (no markdown):
{{"verdict": "fake" or "real", "confidence": 0.0-1.0, "explanation": "one sentence", "red_flags": []}}"""

        for attempt in range(2):
            try:
                response = client.models.generate_content(model="gemini-2.0-flash-lite", contents=prompt)
                break
            except Exception as e:
                if "429" in str(e) and attempt == 0:
                    time.sleep(10)
                    continue
                raise

        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = _json.loads(text.strip())
        return {
            "available": True,
            "verdict": data.get("verdict", "unknown").lower(),
            "confidence": float(data.get("confidence", 0.5)),
            "explanation": data.get("explanation", ""),
            "red_flags": data.get("red_flags", []),
            "error": None,
            "source": "Gemini-2.0-flash",
        }
    except Exception as e:
        return {"available": False, "verdict": None, "explanation": None, "error": str(e)}


def get_smart_search_query(claim: str, api_key: str) -> str:
    """Use Groq to generate an optimized search query for the claim."""
    if not api_key or api_key.startswith("YOUR_"):
        return claim
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        normalized = claim.strip().capitalize()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f'Generate a short Google search query (max 8 words) to find fact-checks or news about this claim: "{normalized}". Reply with ONLY the search query, nothing else.'
            }],
            temperature=0.0,
            max_tokens=30,
        )
        return response.choices[0].message.content.strip().strip('"')
    except Exception:
        return claim


def get_groq_related_links(claim: str, api_key: str, tavily_key: str = "", smart_query: str = "") -> list:
    """Search for real fact-check links using Tavily. Falls back to search URLs."""
    normalized = claim.strip().capitalize()
    search_query = f"fact check {smart_query}" if smart_query else f"fact check {normalized}"

    # Try Tavily first — returns real, live article URLs
    if tavily_key and not tavily_key.startswith("YOUR_"):
        try:
            from tavily import TavilyClient
            client = TavilyClient(tavily_key)
            results = client.search(search_query, max_results=5, search_depth="basic")
            links = []
            for r in results.get("results", []):
                url = r.get("url", "")
                domain = urllib.parse.urlparse(url).netloc.lstrip("www.")
                is_fc = domain in FACT_CHECK_DOMAINS
                is_cred = domain in CREDIBLE_DOMAINS or is_fc
                links.append({
                    "title": r.get("title", domain),
                    "url": url,
                    "snippet": r.get("content", "")[:200],
                    "domain": domain,
                    "is_fact_check": is_fc,
                    "is_credible": is_cred,
                })
            if links:
                return links
        except Exception:
            pass

    # Fallback — working search URLs
    query = urllib.parse.quote(normalized)
    return [
        {
            "title": f"Snopes: {normalized}",
            "url": f"https://www.snopes.com/search/{query}/",
            "snippet": "Search Snopes fact-check database for this claim",
            "domain": "snopes.com",
            "is_fact_check": True,
            "is_credible": True,
        },
        {
            "title": f"PolitiFact: {normalized}",
            "url": f"https://www.politifact.com/search/?q={query}",
            "snippet": "Search PolitiFact for fact-checks on this claim",
            "domain": "politifact.com",
            "is_fact_check": True,
            "is_credible": True,
        },
        {
            "title": f"FactCheck.org: {normalized}",
            "url": f"https://www.factcheck.org/?s={query}",
            "snippet": "Search FactCheck.org for this claim",
            "domain": "factcheck.org",
            "is_fact_check": True,
            "is_credible": True,
        },
    ]





def aggregate_verdict(
    heuristic_result: dict,
    google_fc: dict,
    claimbuster: dict,
    newsapi: dict,
    related_links: dict,
    gemini: dict = None,
    groq: dict = None,
    claim_text: str = "",
) -> dict:
    """Combine all signals into a final verdict with confidence score."""
    score = 0.0
    signals = []

    heuristic_prob = heuristic_result.get("fake_probability", 0.5)
    gravity_fired = heuristic_prob >= 0.55

    # Gather cross-reference counts up front — used to override other signals
    credible_count = newsapi.get("credible_count", 0) if newsapi.get("available") else 0
    fc_true  = sum(1 for r in google_fc.get("results", []) if r.get("verdict") == "true")
    fc_false = google_fc.get("flagged_false", 0) if google_fc.get("available") else 0

    # ── HARD OVERRIDE: if 3+ credible sources cover this claim, it's almost certainly real ──
    # This beats gravity, Google FC noise, and everything else
    if credible_count >= 3:
        final_score = max(0.05, min(0.35, 0.30 - (credible_count - 3) * 0.03))
        prediction = "real"
        confidence = 1.0 - final_score
        signals.append(f"Cross-reference: {credible_count} credible sources confirm this story")
        if gravity_fired:
            signals.append("Gravity check overridden — claim confirmed by credible sources")
        return _build_verdict_result(prediction, confidence, final_score, signals,
                                     google_fc, newsapi, related_links, claimbuster,
                                     claim_text=claim_text)

    # 1. AI verdict — Groq first, Gemini as fallback
    ai_result = None
    if groq and groq.get("available"):
        ai_result = groq
    elif gemini and gemini.get("available"):
        ai_result = gemini

    # ── HARD OVERRIDE: if AI is highly confident this is REAL, trust it over stale FC ──
    if ai_result and ai_result.get("verdict") == "real" and ai_result.get("confidence", 0) >= 0.70:
        final_score = 0.05
        prediction = "real"
        confidence = 1.0 - final_score
        source = ai_result.get("source", "AI")
        signals.append(f"AI Analysis ({source}): {ai_result.get('explanation', 'Verified as real')}")
        signals.append(f"AI override: high-confidence real verdict supersedes other signals")
        return _build_verdict_result(prediction, confidence, final_score, signals,
                                     google_fc, newsapi, related_links, claimbuster,
                                     claim_text=claim_text)

    # ── HARD OVERRIDE: if AI is highly confident this is FAKE, lock it in ──
    if ai_result and ai_result.get("verdict") == "fake" and ai_result.get("confidence", 0) >= 0.95:
        final_score = 0.97
        prediction = "fake"
        confidence = final_score
        source = ai_result.get("source", "AI")
        signals.append(f"AI Analysis ({source}): {ai_result.get('explanation', 'Flagged as fake')}")
        if ai_result.get("red_flags"):
            signals.append(f"Red flags: {', '.join(ai_result['red_flags'][:3])}")
        return _build_verdict_result(prediction, confidence, final_score, signals,
                                     google_fc, newsapi, related_links, claimbuster,
                                     claim_text=claim_text)

    if ai_result:
        ai_verdict = ai_result.get("verdict", "unknown")
        ai_conf = ai_result.get("confidence", 0.5)
        source = ai_result.get("source", "AI")
        if ai_verdict == "fake":
            score += ai_conf * 0.50
            signals.append(f"AI Analysis ({source}): {ai_result.get('explanation', 'Flagged as fake')}")
            if ai_result.get("red_flags"):
                signals.append(f"Red flags: {', '.join(ai_result['red_flags'][:3])}")
        elif ai_verdict == "real":
            score -= ai_conf * 0.15
            signals.append(f"AI Analysis ({source}): {ai_result.get('explanation', 'Appears credible')}")

    # 2. Google Fact Check — only count if flagged false AND no credible sources confirm it
    fc_stale = _fc_results_are_stale(google_fc.get("results", []), newsapi)

    if google_fc.get("available") and google_fc.get("total_checks", 0) > 0:
        total = google_fc.get("total_checks", 1)
        if fc_false > 0 and fc_stale:
            # Stale fact-checks — the claim was false then but confirmed true now by recent news
            signals.append(f"Fact-check APIs: {fc_false}/{total} old results flagged false (superseded by recent credible coverage)")
        elif fc_false > 0 and credible_count == 0:
            # Require 2+ false results before treating as strong fake signal
            if fc_false >= 2:
                fc_score = 0.55 + min(fc_false / total, 1.0) * 0.20
                score = max(score, fc_score)
                signals.append(f"Fact-check APIs: {fc_false}/{total} results flagged as false")
            else:
                score += 0.10
                signals.append(f"Fact-check APIs: {fc_false}/{total} result flagged as false (weak signal)")
        elif fc_false > 0 and credible_count > 0:
            score += 0.10
            signals.append(f"Fact-check APIs: {fc_false}/{total} flagged (conflicting with {credible_count} credible source(s))")
        elif fc_true > 0:
            score -= 0.15
            signals.append(f"Fact-check APIs: claim verified true by fact-checkers")
        else:
            signals.append(f"Fact-check APIs: {total} checks found, none conclusive")
    elif google_fc.get("available"):
        signals.append("Fact-check APIs: no existing fact-checks found")

    # 3. ClaimBuster
    if claimbuster.get("available") and claimbuster.get("check_worthy"):
        score += 0.10
        signals.append(f"ClaimBuster: claim is check-worthy (score {claimbuster.get('score', 0):.2f})")

    # 4. NewsAPI cross-reference
    if newsapi.get("available"):
        total_articles = len(newsapi.get("articles", []))
        if credible_count == 0 and total_articles == 0:
            score += 0.15
            signals.append("Cross-reference: no news coverage found for this claim")
        elif credible_count == 0:
            score += 0.20
            signals.append("Cross-reference: 0 credible sources found covering this story")
        elif credible_count <= 2:
            score += 0.05
            signals.append(f"Cross-reference: only {credible_count} credible source(s) found")

    # 5. Heuristic / gravity
    ai_available = (groq and groq.get("available")) or (gemini and gemini.get("available"))
    if gravity_fired:
        if fc_false > 0 and credible_count == 0:
            score = max(score, heuristic_prob)
            if not ai_available:
                signals.append(f"Heuristic analysis: {heuristic_prob:.0%} fake probability (gravity + fact-check confirmed)")
        elif credible_count >= 1:
            score += heuristic_prob * 0.10
            if not ai_available:
                signals.append(f"Heuristic analysis: {heuristic_prob:.0%} fake probability (partially offset by {credible_count} source(s))")
        else:
            score = max(score, heuristic_prob)
            if not ai_available:
                signals.append(f"Heuristic analysis: {heuristic_prob:.0%} fake probability (unconfirmed extraordinary claim)")
    else:
        score += heuristic_prob * 0.20
        if heuristic_prob > 0.3 and not ai_available:
            signals.append(f"Heuristic analysis: {heuristic_prob:.0%} fake probability")

    final_score = max(0.0, min(1.0, score))
    prediction = "fake" if final_score >= 0.50 else "real"
    confidence = final_score if prediction == "fake" else 1.0 - final_score

    return _build_verdict_result(prediction, confidence, final_score, signals,
                                 google_fc, newsapi, related_links, claimbuster,
                                 claim_text=claim_text)


def _build_verdict_result(prediction, confidence, final_score, signals,
                           google_fc, newsapi, related_links, claimbuster=None,
                           claim_text: str = "") -> dict:

    # Build related links list — Tavily links take priority, skip Google FC if Tavily has results
    all_links = []
    tavily_links = [l for l in related_links.get("links", [])]
    
    # Only add Google FC links if no Tavily links will be injected later
    # (Tavily links are injected in predict.py after this function)
    for link in tavily_links:
        all_links.append(link)
    # Deduplicate by URL
    seen_urls = set()
    deduped_links = []
    for link in all_links:
        if link["url"] and link["url"] not in seen_urls:
            seen_urls.add(link["url"])
            deduped_links.append(link)

    # Sort: fact-checks first, then credible, then rest
    deduped_links.sort(key=lambda x: (
        0 if x.get("is_fact_check") else (1 if x.get("is_credible") else 2)
    ))

    # Filter out irrelevant fact-check links — only keep if snippet/title shares
    # at least one meaningful word with the original claim
    claim_words = set(re.sub(r'[^a-z\s]', '', claim_text.lower()).split()) - {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'has', 'have',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
        'might', 'shall', 'can', 'to', 'of', 'in', 'on', 'at', 'by', 'for',
        'with', 'about', 'as', 'into', 'through', 'and', 'or', 'but', 'not',
    }
    def _is_relevant(link):
        text = (link.get('title', '') + ' ' + link.get('snippet', '')).lower()
        return any(w in text for w in claim_words if len(w) > 3)

    deduped_links = [l for l in deduped_links if _is_relevant(l)]

    return {
        "prediction": prediction,
        "confidence_score": round(confidence, 4),
        "fake_probability": round(final_score, 4),
        "signals": signals,
        "related_links": deduped_links[:10],
        "fact_check_results": google_fc.get("results", []),
        "cross_reference_count": newsapi.get("credible_count", 0),
        "claimbuster_score": claimbuster.get("score") if claimbuster else None,
    }


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def run_fact_check(claim: str, api_keys: dict) -> dict:
    """
    Run all fact-check APIs and return aggregated result.

    api_keys dict expected keys:
        google_factcheck_api_key
        claimbuster_api_key
        newsapi_key
        google_custom_search_api_key
        google_custom_search_cx
        gemini_api_key
    """
    google_fc = check_google_factcheck(
        claim, api_keys.get("google_factcheck_api_key", "")
    )
    claimbuster = check_claimbuster(
        claim, api_keys.get("claimbuster_api_key", "")
    )
    newsapi = search_newsapi(
        claim, api_keys.get("newsapi_key", "")
    )
    related = search_related_links(
        claim,
        api_keys.get("google_custom_search_api_key", ""),
        api_keys.get("google_custom_search_cx", ""),
    )
    gemini = check_gemini(
        claim, api_keys.get("gemini_api_key", "")
    )
    groq_result = check_groq(claim, api_keys.get("groq_api_key", ""))

    return {
        "google_factcheck": google_fc,
        "claimbuster": claimbuster,
        "newsapi": newsapi,
        "related_links_raw": related,
        "gemini": gemini,
        "groq": groq_result,
    }
