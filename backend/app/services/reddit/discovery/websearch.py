import re
import time

from ddgs import DDGS

from app.services.reddit.discovery.base import DiscoverySource

SUBREDDIT_URL_RE = re.compile(r"reddit\.com/r/([A-Za-z0-9_]+)", re.IGNORECASE)

EXCLUDED = {
    "search", "user", "users", "comments", "login",
    "register", "about", "premium", "advertising", "settings", "wiki",
}

MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 2


class WebSearchDiscoverySource(DiscoverySource):

    def __init__(self, backend: str = "auto", max_retries: int = MAX_RETRIES):
        self.backend = backend
        self.max_retries = max_retries

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

    def discover_communities(self, query: str, limit: int = 10) -> list[str]:
        search_query = f"site:reddit.com/r/ {query}"
        results = self._search(search_query, max_results=limit * 4)

        communities = []
        seen = set()

        for result in results:
            url = result.get("href") or result.get("url", "")
            match = SUBREDDIT_URL_RE.search(url)
            if not match:
                continue
            subreddit = match.group(1)
            if subreddit.lower() in EXCLUDED:
                continue
            if subreddit not in seen:
                seen.add(subreddit)
                communities.append(subreddit)
            if len(communities) >= limit:
                break

        return communities