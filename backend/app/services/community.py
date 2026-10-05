import logging
import time
from dataclasses import asdict, is_dataclass

import requests

from app.services.reddit.discovery.client import get_discovery_source
from app.services.reddit.client import get_reddit_source
from app.services.community_filter import CommunityFilter
from app.services.ranking import CommunityCandidate, CommunityRanker
from app.services.query_extraction import extract_search_queries


REMOVED_MARKERS = {
    "[removed]",
    "[deleted]",
}

MAX_POST_AGE_SECONDS = 90 * 24 * 60 * 60  # 90 days
MAX_COMMUNITY_CANDIDATES = 15

logger = logging.getLogger(__name__)


def _is_usable_post(post) -> bool:
    title = (getattr(post, "title", "") or "").strip().lower()
    body = (getattr(post, "body", "") or "").strip().lower()

    if title in REMOVED_MARKERS or body in REMOVED_MARKERS:
        return False

    if not title and not body:
        return False

    created = getattr(post, "created_utc", None)

    if created is not None:
        try:
            age_seconds = time.time() - float(created)

            if age_seconds > MAX_POST_AGE_SECONDS:
                return False

        except (TypeError, ValueError):
            pass

    return True


def _belongs_to_subreddit(post, subreddit: str) -> bool:
    post_subreddit = getattr(post, "subreddit", None)

    if not isinstance(post_subreddit, str):
        return False

    return post_subreddit.casefold() == subreddit.casefold()


class CommunityService:

    def __init__(self):
        self.discovery = get_discovery_source()
        self.reddit = get_reddit_source()
        self.ranker = CommunityRanker()
        self.community_filter = CommunityFilter(self.ranker.embeddings)

    def analyze(
        self,
        description: str,
        candidate_limit: int = 10,
        result_limit: int = 5,
        posts_per_community: int = 10,
        trace: dict | None = None,
    ):
        queries = extract_search_queries(description)

        if trace is not None:
            trace.clear()
            trace.update({
                "description": description,
                "queries": queries,
                "discovery": [],
                "discovered_communities": [],
                "bounded_communities": [],
                "retrieval": [],
                "ranker_input": [],
                "community_filter": {},
                "community_ranker": {},
                "ranked_output": [],
                "returned_output": [],
                "providers": {
                    "discovery": type(self.discovery).__name__,
                    "reddit": type(self.reddit).__name__,
                },
                "limits": {
                    "candidate_limit_per_query": candidate_limit,
                    "max_unique_communities": MAX_COMMUNITY_CANDIDATES,
                    "posts_per_community": posts_per_community,
                    "retrieval_limit_per_community": posts_per_community * 2,
                    "result_limit": result_limit,
                    "max_post_age_seconds": MAX_POST_AGE_SECONDS,
                },
            })
            embedding_model = getattr(
                getattr(getattr(self.ranker, "embeddings", None), "model", None),
                "name_or_path",
                None,
            )
            trace["embedding_model"] = embedding_model

        # ---------------------------------------------------------
        # 1. Discover candidate communities
        # ---------------------------------------------------------
        community_queries = {}
        community_names = set()

        for query in queries:
            discovery_started = time.perf_counter() if trace is not None else None
            try:
                subreddits = self.discovery.discover_communities(
                    query,
                    limit=candidate_limit,
                )

            except Exception as exc:
                if trace is not None:
                    trace["discovery"].append({
                        "query": query,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "subreddits": [],
                        "elapsed_seconds": time.perf_counter() - discovery_started,
                    })
                logger.warning(
                    "Community discovery failed for query '%s': %s",
                    query,
                    exc,
                )
                continue

            if trace is not None:
                trace["discovery"].append({
                    "query": query,
                    "status": "success",
                    "subreddits": list(subreddits) if isinstance(subreddits, list) else [],
                    "elapsed_seconds": time.perf_counter() - discovery_started,
                })

            for subreddit in subreddits:
                if not isinstance(subreddit, str):
                    continue

                subreddit = subreddit.strip()

                if not subreddit:
                    continue

                key = subreddit.casefold()

                if key not in community_names:
                    community_names.add(key)
                    community_queries[subreddit] = query

                    logger.debug(
                        "Discovered r/%s from community query '%s'",
                        subreddit,
                        query,
                    )

        logger.info(
            "Discovered %d unique communities",
            len(community_queries),
        )

        # ---------------------------------------------------------
        # 2. Retrieve recent posts for candidate communities
        # ---------------------------------------------------------
        candidates = []

        bounded_subreddits = list(community_queries)[:MAX_COMMUNITY_CANDIDATES]

        if trace is not None:
            trace["discovered_communities"] = [
                {"subreddit": subreddit, "first_discovery_query": query}
                for subreddit, query in community_queries.items()
            ]
            trace["bounded_communities"] = bounded_subreddits

        for subreddit in bounded_subreddits:
            retrieval_trace = {
                "subreddit": subreddit,
                "requested_limit": posts_per_community * 2,
            }
            retrieval_started = time.perf_counter() if trace is not None else None
            try:
                raw_posts = self.reddit.search_posts(
                    subreddit=subreddit,
                    limit=posts_per_community * 2,
                )

            except requests.RequestException as exc:
                if trace is not None:
                    retrieval_trace.update({
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "retrieved_count": 0,
                        "usable_before_candidate_limit": [],
                        "ranker_input_ids": [],
                        "elapsed_seconds": time.perf_counter() - retrieval_started,
                    })
                    trace["retrieval"].append(retrieval_trace)
                logger.warning(
                    "Skipping r/%s after Reddit retrieval failure: %s",
                    subreddit,
                    exc,
                )
                continue

            except Exception as exc:
                if trace is not None:
                    retrieval_trace.update({
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "retrieved_count": 0,
                        "usable_before_candidate_limit": [],
                        "ranker_input_ids": [],
                        "elapsed_seconds": time.perf_counter() - retrieval_started,
                    })
                    trace["retrieval"].append(retrieval_trace)
                logger.warning(
                    "Skipping r/%s after unexpected retrieval failure: %s",
                    subreddit,
                    exc,
                )
                continue

            usable_before_candidate_limit = [
                post
                for post in raw_posts
                if _is_usable_post(post)
                and _belongs_to_subreddit(post, subreddit)
            ]
            usable = usable_before_candidate_limit[:posts_per_community]

            if trace is not None:
                retrieval_trace.update({
                    "status": "success",
                    "retrieved_count": len(raw_posts),
                    "retrieved_posts": [_serialize_post(post) for post in raw_posts],
                    "usable_before_candidate_limit": [
                        _serialize_post(post)
                        for post in usable_before_candidate_limit
                    ],
                    "usable_before_candidate_limit_count": len(usable_before_candidate_limit),
                    "ranker_input_ids": [getattr(post, "id", None) for post in usable],
                    "ranker_input_limit": posts_per_community,
                    "elapsed_seconds": time.perf_counter() - retrieval_started,
                })
                trace["retrieval"].append(retrieval_trace)

            if not usable:
                logger.debug(
                    "No usable recent posts found for r/%s",
                    subreddit,
                )
                continue

            candidates.append(
                CommunityCandidate(
                    subreddit=subreddit,
                    posts=usable,
                )
            )

            if trace is not None:
                trace["ranker_input"].append({
                    "subreddit": subreddit,
                    "posts": [_serialize_post(post) for post in usable],
                })

        logger.info(
            "Built %d community candidates with usable posts",
            len(candidates),
        )

        # ---------------------------------------------------------
        # 3. Embedding-based community filtering/reranking
        # ---------------------------------------------------------
        if candidates:
            filter_trace = {} if trace is not None else None
            filter_started = time.perf_counter() if trace is not None else None
            candidates = self.community_filter.rerank(
                description,
                candidates,
                queries=queries,
                trace=filter_trace,
            )
            if trace is not None:
                trace["community_filter"] = filter_trace
                trace["community_filter"]["elapsed_seconds"] = time.perf_counter() - filter_started

        logger.info(
            "Community filter returned %d candidates",
            len(candidates),
        )

        # ---------------------------------------------------------
        # 4. Final community ranking
        #
        # IMPORTANT:
        # The current CommunityRanker.rank() implementation does
        # not accept `limit` or `context`. Therefore those arguments
        # must NOT be passed here.
        # ---------------------------------------------------------
        if not candidates:
            return []

        ranker_trace = {} if trace is not None else None
        ranker_started = time.perf_counter() if trace is not None else None
        ranked = self.ranker.rank(
            description,
            candidates,
            trace=ranker_trace,
        )

        if trace is not None:
            trace["community_ranker"] = ranker_trace
            trace["community_ranker"]["elapsed_seconds"] = time.perf_counter() - ranker_started
            trace["ranked_output"] = [
                {
                    "rank": index,
                    "subreddit": result.get("subreddit"),
                    "score": result.get("score"),
                    "status": result.get("status"),
                    "signals": result.get("signals"),
                    "top_post_ids": [
                        getattr(post, "id", None)
                        for post in result.get("top_posts", [])
                    ],
                }
                for index, result in enumerate(ranked, start=1)
            ]

        # ---------------------------------------------------------
        # 5. Return only the requested number of communities
        # ---------------------------------------------------------
        selected = ranked[:result_limit]
        if trace is not None:
            trace["returned_output"] = [
                {
                    "rank": index,
                    "subreddit": result.get("subreddit"),
                    "score": result.get("score"),
                    "status": result.get("status"),
                    "signals": result.get("signals"),
                    "top_post_ids": [
                        getattr(post, "id", None)
                        for post in result.get("top_posts", [])
                    ],
                }
                for index, result in enumerate(selected, start=1)
            ]
        return selected


def _serialize_post(post):
    if hasattr(post, "model_dump"):
        return post.model_dump(mode="json")
    if is_dataclass(post):
        return asdict(post)
    if hasattr(post, "__dict__"):
        return dict(vars(post))
    return {"value": str(post)}
