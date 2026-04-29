"""
Inference module for Fake News Detection.
Combines heuristic/transformer analysis with multi-API fact checking.
"""

import torch
import numpy as np
import sys
import re
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from ml_models.fake_news.model import build_model, SimpleTokenizer


CLICKBAIT_PATTERNS = [
    r"\byou won't believe\b", r"\bshocking\b", r"\bbreaking\b",
    r"\bexplosive\b", r"\bsecret\b", r"\bthey don't want you to know\b",
    r"\bmiracle\b", r"\bcure\b", r"\bhoax\b", r"\bconspiracy\b",
    r"\bfake\b", r"\blie\b", r"\bscam\b", r"\bfraud\b",
    r"\bmicrochip\b", r"\bdeep state\b", r"\bnew world order\b",
    r"\bplandemic\b", r"\bwake up\b", r"\bsheep\b", r"\btruth they hide\b",
    r"\bbill gates\b", r"\b5g\b", r"\bchip\b", r"\bpoison\b",
    r"\bkill\b", r"\bdepopulation\b", r"\bsatan\b", r"\billuminati\b",
    r"\bcauses autism\b", r"\bautism\b", r"\bdangerous vaccine\b",
]

CREDIBILITY_INDICATORS = [
    r"\baccording to\b", r"\bresearch shows\b", r"\bstudies indicate\b",
    r"\bexperts say\b", r"\bofficial\b", r"\bconfirmed\b",
]


def load_model(model_path, tokenizer_path, config=None, device="cpu"):
    model = build_model(config)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    tokenizer = SimpleTokenizer.load(tokenizer_path)
    return model, tokenizer


def fetch_url_text(url):
    """Fetch article text from URL."""
    try:
        import urllib.request
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text_parts = []
                self.skip_tags = {"script", "style", "nav", "footer", "header"}
                self.current_skip = False

            def handle_starttag(self, tag, attrs):
                if tag in self.skip_tags:
                    self.current_skip = True

            def handle_endtag(self, tag):
                if tag in self.skip_tags:
                    self.current_skip = False

            def handle_data(self, data):
                if not self.current_skip:
                    stripped = data.strip()
                    if len(stripped) > 20:
                        self.text_parts.append(stripped)

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        parser = TextExtractor()
        parser.feed(html)
        return " ".join(parser.text_parts[:50])
    except Exception:
        return None


def analyze_text_features(text):
    """Heuristic analysis of text for fake news indicators."""
    text_lower = text.lower()
    clickbait_count = sum(1 for p in CLICKBAIT_PATTERNS if re.search(p, text_lower))
    credibility_count = sum(1 for p in CREDIBILITY_INDICATORS if re.search(p, text_lower))

    exclamation_ratio = text.count("!") / max(len(text), 1)
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

    sentences = re.split(r"[.!?]+", text)
    lengths = [len(s.split()) for s in sentences if s.strip()]
    length_variance = float(np.var(lengths)) if lengths else 0.0

    return {
        "clickbait_indicators": clickbait_count,
        "credibility_indicators": credibility_count,
        "exclamation_ratio": round(exclamation_ratio, 4),
        "caps_ratio": round(caps_ratio, 4),
        "sentence_length_variance": round(length_variance, 4),
        "word_count": len(text.split()),
    }


def _heuristic_only_predict(text: str) -> dict:
    """
    Fallback prediction using only heuristics when model is not trained yet.
    Returns same shape as full predict().
    """
    features = analyze_text_features(text)
    cb = features["clickbait_indicators"]
    cr = features["credibility_indicators"]
    er = features["exclamation_ratio"]
    caps = features["caps_ratio"]

    # Simple scoring: more clickbait + less credibility = higher fake prob
    score = 0.3  # base
    score += min(cb * 0.08, 0.32)
    score -= min(cr * 0.06, 0.18)
    score += min(er * 5, 0.10)
    score += min(caps * 2, 0.10)
    fake_prob = max(0.05, min(0.95, score))

    prediction = "fake" if fake_prob > 0.5 else "real"
    confidence = fake_prob if prediction == "fake" else 1.0 - fake_prob

    anomalies = []
    if cb > 2:
        anomalies.append(f"clickbait language detected ({cb} patterns)")
    if caps > 0.1:
        anomalies.append("excessive capitalization")
    if er > 0.02:
        anomalies.append("excessive exclamation marks")
    if cr == 0 and fake_prob > 0.5:
        anomalies.append("no credibility indicators found")

    return {
        "fake_probability": round(fake_prob, 4),
        "confidence_score": round(confidence, 4),
        "prediction": prediction,
        "suspicious_tokens": [],
        "detected_anomalies": anomalies,
        "text_features": features,
        "model_used": "Heuristic Analyzer (model not trained yet)",
    }


def predict(text_or_url: str, model_path: str, tokenizer_path: str,
            config: dict = None, device: str = "cpu",
            api_keys: dict = None) -> dict:
    """
    Full fake news inference pipeline.
    Accepts raw text or URL.
    Optionally runs multi-API fact checking when api_keys are provided.
    """
    import os
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    # Resolve URL to text
    is_url = text_or_url.startswith("http://") or text_or_url.startswith("https://")
    if is_url:
        text = fetch_url_text(text_or_url)
        if not text:
            return {"error": "Failed to fetch URL content", "content_type": "news"}
        source_url = text_or_url
    else:
        text = text_or_url
        source_url = None

    if len(text.strip()) < 10:
        return {"error": "Text too short for analysis", "content_type": "news"}

    # ── Transformer model (if trained) ──────────────────────────────────────
    model_available = os.path.exists(model_path) and os.path.exists(tokenizer_path)

    if model_available:
        try:
            model, tokenizer = load_model(model_path, tokenizer_path, config, device)
            max_len = config.get("max_length", 512) if config else 512
            ids, mask = tokenizer.encode(text, max_len)
            ids_t = torch.tensor([ids], dtype=torch.long).to(device)
            mask_t = torch.tensor([mask], dtype=torch.long).to(device)

            with torch.no_grad():
                prob, attn = model(ids_t, mask_t)

            prob_val = prob.item()
            attn_weights = attn.squeeze(0).mean(0).cpu().numpy()
            words = text.split()[:max_len]
            attn_flat = attn_weights[:len(words)]
            top_indices = np.argsort(attn_flat)[-10:][::-1]
            suspicious_tokens = [words[i] for i in top_indices if i < len(words)]

            text_features = analyze_text_features(text)
            prediction = "fake" if prob_val > 0.5 else "real"
            confidence = prob_val if prob_val > 0.5 else 1.0 - prob_val

            anomalies = []
            if text_features["clickbait_indicators"] > 2:
                anomalies.append(f"clickbait language detected ({text_features['clickbait_indicators']} patterns)")
            if text_features["caps_ratio"] > 0.1:
                anomalies.append("excessive capitalization")
            if text_features["exclamation_ratio"] > 0.02:
                anomalies.append("excessive exclamation marks")
            if text_features["credibility_indicators"] == 0 and prob_val > 0.6:
                anomalies.append("no credibility indicators found")
            if prob_val > 0.7:
                anomalies.append("semantic pattern matches known fake news")

            heuristic_result = {
                "fake_probability": round(prob_val, 4),
                "confidence_score": round(confidence, 4),
                "prediction": prediction,
                "suspicious_tokens": suspicious_tokens,
                "detected_anomalies": anomalies,
                "text_features": text_features,
                "model_used": "FakeNewsModel (Custom Transformer)",
            }
        except Exception as e:
            heuristic_result = _heuristic_only_predict(text)
            heuristic_result["model_warning"] = f"Model load failed: {e}"
    else:
        heuristic_result = _heuristic_only_predict(text)

    # ── Multi-API Fact Checking ──────────────────────────────────────────────
    fact_check_data = {}
    related_links = []
    signals = []
    cross_reference_count = 0
    fact_check_results = []
    claimbuster_score = None

    if api_keys:
        from backend.services.news_fact_checker import (
            run_fact_check, aggregate_verdict,
            get_smart_search_query, get_groq_related_links, check_groq
        )

        claim = text[:500]
        # Use Groq to generate a smarter search query for search APIs
        smart_query = get_smart_search_query(claim, api_keys.get("groq_api_key", ""))
        search_claim = smart_query if smart_query and smart_query != claim else claim

        # Run search APIs with smart query, but Groq verdict must use original claim
        raw = run_fact_check(search_claim, api_keys)
        raw["groq"] = check_groq(claim, api_keys.get("groq_api_key", ""))
        aggregated = aggregate_verdict(
            heuristic_result,
            raw["google_factcheck"],
            raw["claimbuster"],
            raw["newsapi"],
            raw["related_links_raw"],
            raw.get("gemini"),
            raw.get("groq"),
            claim_text=claim,
        )

        # Override prediction with aggregated verdict
        final_prediction = aggregated["prediction"]
        final_confidence = aggregated["confidence_score"]
        final_fake_prob = aggregated["fake_probability"]
        related_links = aggregated["related_links"]
        signals = aggregated["signals"]
        cross_reference_count = aggregated["cross_reference_count"]
        fact_check_results = aggregated["fact_check_results"]
        claimbuster_score = aggregated["claimbuster_score"]
        fact_check_data = {
            "google_factcheck_available": raw["google_factcheck"].get("available", False),
            "claimbuster_available": raw["claimbuster"].get("available", False),
            "newsapi_available": raw["newsapi"].get("available", False),
        }

        # Inject Groq verdict as first link + Groq-suggested relevant links
        groq_r = raw.get("groq", {})
        # Don't add groq.com as a link — it's shown in the AI verdict card already
        groq_links = get_groq_related_links(claim, api_keys.get("groq_api_key", ""), api_keys.get("tavily_api_key", ""), smart_query=smart_query)
        related_links.extend(groq_links)

        # Update anomalies with AI red flags
        if groq_r.get("available") and groq_r.get("verdict") == "fake":
            ai_anomalies = []
            if groq_r.get("explanation"):
                ai_anomalies.append(groq_r["explanation"])
            ai_anomalies.extend(groq_r.get("red_flags", []))
            heuristic_result["detected_anomalies"] = ai_anomalies + heuristic_result["detected_anomalies"]
        if groq_r.get("available") and groq_r.get("explanation"):
            heuristic_result["possible_generation_method"] = groq_r["explanation"]
    else:
        final_prediction = heuristic_result["prediction"]
        final_confidence = heuristic_result["confidence_score"]
        final_fake_prob = heuristic_result["fake_probability"]

    return {
        "content_type": "news",
        "prediction": final_prediction,
        "confidence_score": round(final_confidence, 4),
        "fake_probability": round(final_fake_prob, 4),
        "source_url": source_url,
        "word_count": heuristic_result["text_features"]["word_count"],
        "detected_anomalies": heuristic_result["detected_anomalies"],
        "suspicious_tokens": heuristic_result.get("suspicious_tokens", []),
        "text_features": heuristic_result["text_features"],
        "possible_generation_method": heuristic_result.get("possible_generation_method", "AI-generated or human-written misinformation"),
        "model_used": heuristic_result["model_used"],
        "text_preview": text[:300] + "..." if len(text) > 300 else text,
        # Fact-check enrichment
        "related_links": related_links,
        "fact_check_results": fact_check_results,
        "signals": signals,
        "cross_reference_count": cross_reference_count,
        "claimbuster_score": claimbuster_score,
        "fact_check_apis": fact_check_data,
    }


if __name__ == "__main__":
    import json
    if len(sys.argv) < 4:
        print("Usage: python predict.py <text_or_url> <model_path> <tokenizer_path>")
        sys.exit(1)
    result = predict(sys.argv[1], sys.argv[2], sys.argv[3])
    print(json.dumps(result, indent=2))
