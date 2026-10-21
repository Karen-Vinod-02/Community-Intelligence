from concurrent.futures import ThreadPoolExecutor
import logging
import re
from urllib.parse import urlparse

import requests
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from app.models.analysis import CompetitorMention
from app.services.embeddings import get_embedding_service
from app.services.generation import expand_search_terms
from app.services.query_extraction import (
    PRODUCT_WRAPPER_WORDS,
    extract_problem_hypotheses,
    extract_search_queries,
)
from app.services.web_search import search_text


_EXCLUDED_DOMAINS = {
    "reddit.com",
    "linkedin.com",
    "facebook.com",
    "youtube.com",
    "wikipedia.org",
    "archive.org",
    "news.ycombinator.com",
    "bestmoney.com",
    "nerdwallet.com",
    "forbes.com",
    "businessinsider.com",
    "investopedia.com",
    "bankrate.com",
    "money.com",
    "theverge.com",
    "techcrunch.com",
    "wired.com",
}
_EDITORIAL_PATH = re.compile(
    r"/(?:articles?|blog|guides?|reviews?|resources?|news|learn|"
    r"press|stories|research|insights|learn-more)(?:[-/]|$)",
    re.IGNORECASE,
)
_EDITORIAL_TITLE = re.compile(
    r"\b(?:how to|guide|best\b.{0,40}\bapps?\b|top \d+|"
    r"review|tips|freshman guide|comparison|versus|vs\.?)\b",
    re.IGNORECASE,
)
_PRODUCT_COPY = re.compile(
    r"\b(?:app|software|platform|product|tool|service|"
    r"download|sign up|pricing|subscription|book|hire|marketplace)\b",
    re.IGNORECASE,
)
_SHOW_HN_TITLE = re.compile(r"^show hn:", re.IGNORECASE)
_PRODUCT_DIRECTORIES = {
    "producthunt.com",
    "alternativeto.net",
    "g2.com",
    "capterra.com",
    "getapp.com",
}
logger = logging.getLogger(__name__)


def _search(query: str) -> list[dict]:
    results, _, error = search_text(query, max_results=8, timeout=5)
    if error is not None:
        raise RuntimeError("Product search failed on all indexed-search backends.") from error
    return results


def _search_hacker_news(query: str) -> list[dict]:
    response = requests.get(
        "https://hn.algolia.com/api/v1/search",
        params={"query": query, "tags": "story", "hitsPerPage": 8},
        timeout=(1.5, 3),
    )
    response.raise_for_status()
    payload = response.json()
    hits = payload.get("hits", [])
    results = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        title = str(hit.get("title") or "").strip()
        url = str(hit.get("url") or "").strip()
        if not title or not _SHOW_HN_TITLE.match(title) or not url:
            continue
        results.append({
            "href": url,
            "title": title,
            "body": str(hit.get("story_text") or title),
        })
    return results


def _normalise_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _tokens(text: str) -> set[str]:
    return {
        _normalise_token(token)
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) > 2
        and token not in ENGLISH_STOP_WORDS
        and token not in PRODUCT_WRAPPER_WORDS
    }


def _problem_terms(description: str, extra_phrases: list[str] | None = None) -> set[str]:
    hypotheses = extract_problem_hypotheses(description)
    if hypotheses:
        source = " ".join(
            " ".join((
                hypothesis.actor or "",
                hypothesis.action or "",
                hypothesis.target or "",
            ))
            for hypothesis in hypotheses
        )
    else:
        source = description
    terms = _tokens(source)
    # Synonym phrases widen the match vocabulary (e.g. househelp -> maid, housekeeper)
    for phrase in extra_phrases or []:
        terms |= _tokens(phrase)
    return terms


def _is_product_page(url: str, title: str, host: str) -> bool:
    path = urlparse(url).path.rstrip("/")
    if any(
        host == domain or host.endswith(f".{domain}")
        for domain in _EXCLUDED_DOMAINS
    ):
        return False
    if _EDITORIAL_PATH.search(path) or _EDITORIAL_TITLE.search(title):
        return False

    domain = host.removeprefix("www.")
    if domain in _PRODUCT_DIRECTORIES and not path:
        return False
    if domain == "producthunt.com" and not re.match(
        r"/products/[^/]+$", path, re.IGNORECASE
    ):
        return False
    if domain == "alternativeto.net" and not re.match(
        r"/software/[^/]+", path, re.IGNORECASE
    ):
        return False
    if domain in {"g2.com", "capterra.com", "getapp.com"} and not re.search(
        r"/(?:products?|software)/[^/]+", path, re.IGNORECASE
    ):
        return False
    return True


def find_competitors(description: str, limit: int = 5) -> list[CompetitorMention]:
    problem_queries = extract_search_queries(description)
    if not problem_queries:
        return []

    extra_phrases = expand_search_terms(description, max_terms=3, timeout=6.0)
    problem_terms = _problem_terms(description, extra_phrases)
    if not problem_terms:
        problem_terms = _tokens(problem_queries[0])
    if not problem_terms:
        return []

    core_query = problem_queries[0]
    queries = [
        f'site:producthunt.com/products "{core_query}"',
        f'site:alternativeto.net/software "{core_query}"',
        f'"{core_query}" app product software',
        f'"{core_query}" alternatives product',
    ]
    for phrase in extra_phrases[:2]:
        queries.append(f"{phrase} app")
        queries.append(f"{phrase} platform pricing")

    with ThreadPoolExecutor(max_workers=len(queries) + 1) as executor:
        futures = [executor.submit(_search, query) for query in queries]
        hacker_news_future = executor.submit(
            _search_hacker_news,
            f"{core_query} app OR alternative OR Show HN",
        )
        results = []
        search_failures = 0
        for query, future in zip(queries, futures, strict=True):
            try:
                results.append(future.result())
            except Exception:
                logger.warning("Competitor web search failed for query %r", query, exc_info=True)
                search_failures += 1
                results.append([])
        try:
            results.append(hacker_news_future.result())
        except Exception as exc:
            logger.info("Hacker News competitor search skipped: %s", exc)
            results.append([])

    if search_failures == len(queries):
        raise RuntimeError("All competitor search sources are unavailable.")

    candidates = []
    seen_urls = set()
    for result_set in results:
        for result in result_set:
            url = result.get("href") or result.get("url")
            if not url:
                continue
            parsed_url = urlparse(url)
            host = (parsed_url.hostname or "").lower().removeprefix("www.")
            if not host or any(
                host == domain or host.endswith(f".{domain}")
                for domain in _EXCLUDED_DOMAINS
            ):
                continue

            title = (result.get("title") or "").strip()
            snippet = " ".join(
                str(result.get(key) or "").strip()
                for key in ("body", "snippet", "description")
            ).strip()
            if (
                not title
                or not snippet
                or not _is_product_page(url, title, host)
            ):
                continue
            is_product_directory = host in _PRODUCT_DIRECTORIES
            if not is_product_directory and not _PRODUCT_COPY.search(
                f"{title} {snippet}"
            ):
                continue
            url_key = f"{host}{parsed_url.path.rstrip('/')}".casefold()
            if url_key in seen_urls:
                continue
            seen_urls.add(url_key)
            text_terms = _tokens(f"{title} {snippet}")
            overlap = len(problem_terms & text_terms) / len(problem_terms)
            candidates.append((title, snippet, url, host, overlap))

    if not candidates:
        return []

    embeddings = get_embedding_service()
    description_vector = embeddings.encode([description])[0]
    texts = [f"{title}. {snippet}" for title, snippet, _, _, _ in candidates]
    semantic_scores = [
        max(0.0, min(1.0, float(score)))
        for score in embeddings.encode(texts) @ description_vector
    ]

    scored = []
    for (title, snippet, url, host, overlap), semantic in zip(
        candidates, semantic_scores, strict=True
    ):
        relevance = 0.5 * semantic + 0.5 * min(1.0, overlap)
        scored.append((title, snippet, url, host, overlap, semantic, relevance))
    scored.sort(key=lambda item: item[6], reverse=True)

    strict_overlap = max(0.5, min(1.0, 2 / len(problem_terms)))
    selected = []
    used = set()

    # Pass 1: strict
    for item in scored:
        title, snippet, url, host, overlap, semantic, relevance = item
        if overlap >= strict_overlap and relevance >= 0.55:
            selected.append(item)
            used.add(url)
        if len(selected) >= limit:
            break

    # Pass 2: relaxed, so the user always gets plausible products
    if len(selected) < limit:
        for item in scored:
            title, snippet, url, host, overlap, semantic, relevance = item
            if url in used:
                continue
            if relevance >= 0.35 and (overlap >= 0.2 or semantic >= 0.45):
                selected.append(item)
                used.add(url)
            if len(selected) >= limit:
                break

    return [
        CompetitorMention(
            name=title[:180],
            description=snippet[:500],
            evidence=snippet[:500],
            source_url=url,
            source_platform=host,
            relevance=relevance,
        )
        for title, snippet, url, host, _, _, relevance in selected
    ]