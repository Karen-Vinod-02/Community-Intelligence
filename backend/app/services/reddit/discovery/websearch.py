import logging
import re

import requests

from app.services.embeddings import get_embedding_service
from app.services.reddit.discovery.base import (
    CommunityDiscoveryResults,
    DiscoverySource,
)
from app.services.web_search import search_text


SUBREDDIT_URL_RE = re.compile(
    r"reddit\.com/r/([A-Za-z0-9_]+)",
    re.IGNORECASE,
)

EXCLUDED = {
    "search",
    "user",
    "users",
    "comments",
    "login",
    "register",
    "about",
    "premium",
    "advertising",
    "settings",
    "wiki",
}

# Retrieve a larger candidate pool from web search so that useful communities are not lost before the downstream Reddit ranking stage.
DISCOVERY_RESULT_MULTIPLIER = 6

logger = logging.getLogger(__name__)


class WebSearchDiscoverySource(DiscoverySource):

    def __init__(
        self,
        backend: str = "duckduckgo",
    ):
        self.backend = backend

    def _search(
        self,
        query: str,
        max_results: int,
    ) -> list:
        results, _, error = search_text(
            query,
            max_results=max_results,
            preferred_backend=self.backend,
        )
        if error is not None:
            logger.warning(
                "Reddit community indexed search failed; trying Reddit search API: %s",
                error,
            )
            results = self._search_reddit_api(query, max_results)
            if not results:
                raise RuntimeError(
                    "Reddit community search failed on indexed search and Reddit API."
                ) from error
        return results

    @staticmethod
    def _search_reddit_api(query: str, max_results: int) -> list[dict]:
        """Use Reddit's public post search when web indexes are unavailable."""
        try:
            response = requests.get(
                "https://www.reddit.com/search.json",
                params={
                    "q": re.sub(
                        r"^site:reddit\.com/r/\s*",
                        "",
                        query,
                        flags=re.IGNORECASE,
                    ),
                    "sort": "relevance",
                    "t": "month",
                    "type": "link",
                    "limit": max_results,
                    "raw_json": 1,
                },
                headers={
                    "User-Agent": "CommInt/0.1 (community research prototype)"
                },
                timeout=(2, 6),
            )
            response.raise_for_status()
            payload = response.json()
            results = []
            for child in payload.get("data", {}).get("children", []):
                post = child.get("data", {})
                subreddit = str(post.get("subreddit") or "").strip()
                permalink = str(post.get("permalink") or "").strip()
                if not subreddit or not permalink:
                    continue
                results.append({
                    "href": f"https://www.reddit.com{permalink}",
                    "title": str(post.get("title") or ""),
                    "body": str(post.get("selftext") or ""),
                    "verified_source": "reddit_json",
                })
            return results
        except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
            logger.info("Reddit community API fallback failed: %s", exc)
            return []

    @staticmethod
    def _result_text(result: dict) -> str:
        return " ".join(
            str(result.get(field) or "")
            for field in (
                "title",
                "body",
                "snippet",
                "description",
            )
        ).strip()

    def _relevance_score(
        self,
        query: str,
        result: dict,
    ) -> float:
        result_text = self._result_text(result)

        if not result_text:
            return 0.0

        embeddings = get_embedding_service().encode(
            [
                query,
                result_text,
            ]
        )

        return float(
            embeddings[0] @ embeddings[1]
        )

    def discover_communities(
        self,
        query: str,
        limit: int = 10,
    ) -> list[str]:
        search_query = f"site:reddit.com/r/ {query}"

        results = self._search(
            search_query,
            max_results=limit * DISCOVERY_RESULT_MULTIPLIER,
        )

        communities = []
        seen = set()
        relevance_by_community = {}
        evidence = []

        for result in results:
            url = (
                result.get("href")
                or result.get("url", "")
            )

            match = SUBREDDIT_URL_RE.search(url)
            if not match:
                continue
            subreddit = match.group(1)
            if subreddit.lower() in EXCLUDED:
                continue

            score = self._relevance_score(
                query,
                result,
            )

            key = subreddit.casefold()
            relevance_by_community[key] = max(
                score,
                relevance_by_community.get(key, 0.0),
            )

            evidence.append(
                {
                    "query": query,
                    "subreddit": subreddit,
                    "score": round(score, 3),
                    "url": url,
                    "title": result.get("title") or "",
                    "snippet": (
                        result.get("body")
                        or result.get("snippet")
                        or result.get("description")
                        or ""
                    ),
                }
            )

            if key not in seen:
                seen.add(key)
                communities.append(subreddit)

        communities.sort(
            key=lambda name: relevance_by_community[name.casefold()],
            reverse=True,
        )

        logger.debug(
            "Discovery query '%s' produced %d candidate communities",
            query,
            len(communities),
        )

        selected = communities[:limit]
        selected_keys = {name.casefold() for name in selected}
        evidence_by_community = {}
        for item in evidence:
            if item["subreddit"].casefold() in selected_keys:
                evidence_by_community.setdefault(
                    item["subreddit"],
                    [],
                ).append(item)

        return CommunityDiscoveryResults(selected, evidence_by_community)
