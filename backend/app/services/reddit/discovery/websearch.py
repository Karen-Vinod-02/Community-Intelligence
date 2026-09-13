import logging
import os
import re
import time
from functools import lru_cache

from ddgs import DDGS

from app.services.embeddings import EmbeddingService
from app.services.reddit.discovery.base import DiscoverySource

SUBREDDIT_URL_RE = re.compile(r"reddit\.com/r/([A-Za-z0-9_]+)", re.IGNORECASE)

EXCLUDED = {
    "search", "user", "users", "comments", "login",
    "register", "about", "premium", "advertising", "settings", "wiki",
}

MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 2
DISCOVERY_RELEVANCE_THRESHOLD = float(
    os.getenv("REDDIT_DISCOVERY_RELEVANCE_THRESHOLD", "0.18")
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _embedding_service() -> EmbeddingService:
    return EmbeddingService()


class WebSearchDiscoverySource(DiscoverySource):

    def __init__(self, backend: str = "auto", max_retries: int = MAX_RETRIES):
        self.backend = backend
        self.max_retries = max_retries
        self.discovery_evidence: dict[str, list[dict]] = {}

    def _search(self, query: str, max_results: int) -> list:
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                with DDGS() as ddgs:
                    return list(
                        ddgs.text(query, max_results=max_results, backend=self.backend)
                    )
            except Exception as exc:
                last_exception = exc
                if attempt < self.max_retries:
                    time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))

        print(f"[Discovery] web search failed after retries: {last_exception}")
        return []

    @staticmethod
    def _result_text(result: dict) -> str:
        return " ".join(
            str(result.get(field) or "")
            for field in ("title", "body", "snippet", "description")
        ).strip()

    def _relevance_score(self, query: str, result: dict) -> float:
        result_text = self._result_text(result)
        if not result_text:
            return 0.0

        embeddings = _embedding_service().encode([query, result_text])
        return float(embeddings[0] @ embeddings[1])

    def discover_communities(self, query: str, limit: int = 10) -> list[str]:
        search_query = f"site:reddit.com/r/ {query}"
        results = self._search(search_query, max_results=limit * 4)

        communities = []
        seen = set()
        evidence = []

        for result in results:
            url = result.get("href") or result.get("url", "")
            match = SUBREDDIT_URL_RE.search(url)
            if not match:
                continue
            subreddit = match.group(1)
            if subreddit.lower() in EXCLUDED:
                continue
            score = self._relevance_score(query, result)
            if score < DISCOVERY_RELEVANCE_THRESHOLD:
                logger.debug(
                    "Rejected r/%s for discovery query '%s' (score %.3f): %s",
                    subreddit,
                    query,
                    score,
                    self._result_text(result)[:160],
                )
                continue

            key = subreddit.casefold()
            evidence.append({
                "query": query,
                "subreddit": subreddit,
                "score": round(score, 3),
                "url": url,
                "title": result.get("title") or "",
                "snippet": result.get("body")
                or result.get("snippet")
                or result.get("description")
                or "",
            })
            if key not in seen:
                seen.add(key)
                communities.append(subreddit)
            if len(communities) >= limit:
                break

        for item in evidence:
            self.discovery_evidence.setdefault(item["subreddit"], []).append(item)

        return communities