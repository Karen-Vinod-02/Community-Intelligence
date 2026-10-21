import logging
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from concurrent.futures import ThreadPoolExecutor

from app.models.reddit import RedditPost
from app.services.reddit.discovery.client import get_discovery_source
from app.services.reddit.client import get_reddit_source
from app.services.community_filter import CommunityFilter
from app.services.ranking import CommunityCandidate, CommunityRanker
from app.services.query_extraction import extract_search_queries
from app.services.generation import expand_search_terms
from app.services.web_search import search_text
from app.services.public_communities import (
    _is_relevant_indexed_discussion,
    _verified_reddit_discussion,
    discover_public_communities,
)


REMOVED_MARKERS = {
    "[removed]",
    "[deleted]",
}

MAX_POST_AGE_SECONDS = 90 * 24 * 60 * 60  # 90 days
MAX_COMMUNITY_CANDIDATES = 15
MAX_DISCOVERY_QUERIES = 5
MAX_NETWORK_WORKERS = 4
MAX_REDDIT_RETRIEVAL_WORKERS = 2
MAX_DISCOVERY_FALLBACK_COMMUNITIES = 5
MAX_WEB_THREAD_VERIFICATIONS = 14
PUBLIC_DISCOVERY_EXECUTOR = ThreadPoolExecutor(max_workers=6)

logger = logging.getLogger(__name__)


def _parallel_map(function, items: list, max_workers: int = MAX_NETWORK_WORKERS):
    if not items:
        return []

    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as executor:
        futures = [
            (item, time.perf_counter(), executor.submit(function, item))
            for item in items
        ]
        results = []
        for item, started, future in futures:
            try:
                results.append((item, future.result(), None, time.perf_counter() - started))
            except Exception as exc:
                results.append((item, None, exc, time.perf_counter() - started))
        return results


def _is_usable_post(post) -> bool:
    title = (getattr(post, "title", "") or "").strip().lower()
    body = (getattr(post, "body", "") or "").strip().lower()

    if title in REMOVED_MARKERS or body in REMOVED_MARKERS:
        return False

    if not title and not body:
        return False

    created = getattr(post, "created_utc", None)

    if created is None:
        return False

    try:
        age_seconds = time.time() - float(created)
        if age_seconds > MAX_POST_AGE_SECONDS or age_seconds < -86400:
            return False
        post.recency_verified = True
    except (TypeError, ValueError, AttributeError):
        return False

    return True


def _belongs_to_subreddit(post, subreddit: str) -> bool:
    post_subreddit = getattr(post, "subreddit", None)

    if not isinstance(post_subreddit, str):
        return False

    return post_subreddit.casefold() == subreddit.casefold()


def _merge_queries(*groups: list[str], limit: int) -> list[str]:
    merged = []
    seen = set()
    for group in groups:
        for query in group:
            normalised = " ".join(str(query).split()).casefold()
            if not normalised or normalised in seen:
                continue
            seen.add(normalised)
            merged.append(normalised)
            if len(merged) >= limit:
                return merged
    return merged


def _field(item, name, default=""):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _with_source(post, source: str):
    try:
        return post.model_copy(update={"evidence_source": source})
    except Exception:
        try:
            post.evidence_source = source
        except Exception:
            pass
        return post


def _discovery_evidence_candidates(
    evidence_by_community: dict,
    queries: list[str],
    limit: int,
) -> list[CommunityCandidate]:
    candidates = []
    active_queries = {query.casefold() for query in queries}

    for subreddit, evidence_items in evidence_by_community.items():
        if not isinstance(subreddit, str) or not isinstance(evidence_items, list):
            continue

        posts = []
        seen_urls = set()
        for item in evidence_items:
            if not isinstance(item, dict):
                continue
            if str(item.get("query") or "").casefold() not in active_queries:
                continue

            url = str(item.get("url") or "").strip()
            title = str(item.get("title") or "").strip()
            snippet = str(item.get("snippet") or "").strip()
            if not url.startswith(("https://", "http://")) or not (title or snippet):
                continue
            if url in seen_urls:
                continue
            verified = _verified_reddit_discussion(
                {"href": url},
                recent_days=90,
            )
            if verified is None:
                continue
            seen_urls.add(url)
            verified_subreddit, post = verified
            post.subreddit = verified_subreddit
            posts.append(post)
            if len(posts) >= 3:
                break

        if posts:
            candidates.append(CommunityCandidate(
                subreddit=subreddit,
                posts=posts,
            ))
        if len(candidates) >= limit:
            break

    return candidates


def _web_thread_leads(queries: list[str], max_communities: int = 8) -> list:
    """
    Last-resort evidence: search the web for Reddit threads about the problem,
    verify each thread, and group them by subreddit as thread leads.
    """

    def run(query):
        results, _, error = search_text(
            f"site:reddit.com {query}",
            max_results=8,
            timeout=5,
        )
        if error is not None:
            raise error
        return results

    grouped: dict[str, list] = {}
    seen_urls = set()
    verifications = 0

    for _, results, error, _ in _parallel_map(run, queries[:4]):
        if error is not None or not results:
            continue
        for result in results:
            if verifications >= MAX_WEB_THREAD_VERIFICATIONS:
                break
            url = str(result.get("href") or result.get("url") or "").strip()
            if "reddit.com/r/" not in url or url in seen_urls:
                continue
            seen_urls.add(url)
            verifications += 1
            try:
                verified = _verified_reddit_discussion(
                    {"href": url},
                    recent_days=365,
                )
            except Exception:
                continue
            if verified is None:
                continue
            subreddit, post = verified
            post.subreddit = subreddit
            post = _with_source(post, "web_search_thread")
            bucket = grouped.setdefault(subreddit, [])
            if len(bucket) < 3:
                bucket.append(post)

    leads = []
    for subreddit, posts in grouped.items():
        leads.append((
            f"reddit:{subreddit.casefold()}",
            posts,
            {
                "platform": "reddit",
                "name": subreddit,
                "url": f"https://www.reddit.com/r/{subreddit}/",
                "kind": "community",
            },
        ))
        if len(leads) >= max_communities:
            break
    return leads


def _build_explanation(description: str, result: dict) -> dict:
    posts = result.get("top_posts", []) or []
    name = result.get("name") or result.get("subreddit") or "this community"
    platform = result.get("platform", "reddit")
    prefix = "r/" if platform == "reddit" else ""
    status = result.get("status")
    signals = result.get("signals") or {}
    relevant = signals.get("problem_relevant_posts") or 0

    if status == "thread_lead":
        summary = (
            f"Search surfaced {len(posts)} recent thread(s) in {prefix}{name} where people "
            f"are discussing the problem behind \"{description[:120]}\". Relevance is not "
            "scored yet, so read the threads before posting."
        )
    elif status == "potential":
        summary = (
            f"{prefix}{name} appeared in discovery searches for this problem and has "
            f"{len(posts)} indexed discussion(s) touching it. Treat it as a lower-confidence match."
        )
    else:
        summary = (
            f"{prefix}{name} has {relevant or len(posts)} recent discussion(s) whose "
            "wording and intent match the problem your product solves."
        )

    matches = []
    for post in posts[:3]:
        title = str(_field(post, "title") or "").strip()
        body = str(_field(post, "body") or "").strip()
        text = title
        if body:
            text = f"{title} — {body[:200].strip()}{'…' if len(body) > 200 else ''}"
        if not text:
            continue
        matches.append({
            "label": "Matching discussion",
            "text": text,
            "score": None,
            "source_url": _field(post, "url") or None,
        })

    return {"summary": summary, "matches": matches}


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
        source_status: dict | None = None,
        progress_callback=None,
    ):
        if progress_callback:
            progress_callback("understanding", 5)
        base_queries = extract_search_queries(description)
        expansion = []
        try:
            expansion = PUBLIC_DISCOVERY_EXECUTOR.submit(
                expand_search_terms,
                description,
                4,
                6.0,
            ).result(timeout=9)
        except Exception as exc:
            logger.info("Query expansion skipped: %s", exc)

        queries = _merge_queries(
            base_queries[:3],
            expansion[:2],
            base_queries[3:],
            expansion[2:],
            limit=MAX_DISCOVERY_QUERIES,
        )
        if progress_callback:
            progress_callback("searching", 20)

        public_discovery_future = PUBLIC_DISCOVERY_EXECUTOR.submit(
            discover_public_communities,
            description,
            queries,
        )
        web_leads_future = PUBLIC_DISCOVERY_EXECUTOR.submit(
            _web_thread_leads,
            queries,
        )

        if trace is not None:
            trace.clear()
            trace.update({
                "description": description,
                "queries": queries,
                "discovery": [],
                "public_discovery": {},
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
                    "retrieval_limit_per_community": posts_per_community,
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
        discovery_evidence = {}

        discovery_results = _parallel_map(
            lambda query: self.discovery.discover_communities(
                query,
                limit=candidate_limit,
            ),
            queries,
        )
        if source_status is not None:
            discovery_failures = sum(error is not None for _, _, error, _ in discovery_results)
            source_status["reddit_discovery"] = (
                "unavailable"
                if discovery_results and discovery_failures == len(discovery_results)
                else "partial_failure"
                if discovery_failures
                else "available"
            )
        for query, subreddits, error, elapsed in discovery_results:
            if error is not None:
                if trace is not None:
                    trace["discovery"].append({
                        "query": query,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "subreddits": [],
                        "elapsed_seconds": elapsed,
                    })
                logger.warning(
                    "Community discovery failed for query '%s': %s",
                    query,
                    error,
                )
                continue

            if trace is not None:
                trace["discovery"].append({
                    "query": query,
                    "status": "success",
                    "subreddits": list(subreddits) if isinstance(subreddits, list) else [],
                    "elapsed_seconds": elapsed,
                })

            for evidence_subreddit, items in getattr(
                subreddits,
                "evidence",
                {},
            ).items():
                discovery_evidence.setdefault(
                    evidence_subreddit,
                    [],
                ).extend(items)

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
        if progress_callback:
            progress_callback("communities_found", 40)

        # ---------------------------------------------------------
        # 2. Retrieve recent posts for candidate communities
        # ---------------------------------------------------------
        candidates = []
        indexed_thread_leads = []

        bounded_subreddits = list(community_queries)[:MAX_COMMUNITY_CANDIDATES]

        if trace is not None:
            trace["discovered_communities"] = [
                {"subreddit": subreddit, "first_discovery_query": query}
                for subreddit, query in community_queries.items()
            ]
            trace["bounded_communities"] = bounded_subreddits

        retrieval_results = _parallel_map(
            lambda subreddit: self.reddit.search_posts(
                subreddit=subreddit,
                query=community_queries[subreddit],
                limit=posts_per_community,
            ),
            bounded_subreddits,
            max_workers=MAX_REDDIT_RETRIEVAL_WORKERS,
        )
        retrieval_by_subreddit = {
            subreddit: (raw_posts, error, elapsed)
            for subreddit, raw_posts, error, elapsed in retrieval_results
        }
        retrieval_failures = sum(
            error is not None
            for _, _, error, _ in retrieval_results
        )
        if source_status is not None:
            source_status["reddit_retrieval"] = (
                "unavailable"
                if retrieval_results and retrieval_failures == len(retrieval_results)
                else "no_recent_evidence"
                if not any(raw_posts for _, raw_posts, error, _ in retrieval_results if error is None)
                else "partial_failure"
                if retrieval_failures
                else "available"
            )
        if retrieval_failures:
            failure_counts = Counter(
                (
                    f"HTTP {error.response.status_code}"
                    if getattr(error, "response", None) is not None
                    else type(error).__name__
                )
                for _, _, error, _ in retrieval_results
                if error is not None
            )
            failure_summary = ", ".join(
                f"{reason}: {count}"
                for reason, count in sorted(failure_counts.items())
            )
            logger.info(
                "Reddit results are partial for %d/%d communities "
                "(%s); continuing with public indexed discussion evidence.",
                retrieval_failures,
                len(retrieval_results),
                failure_summary,
            )
        if progress_callback:
            progress_callback("checking_discussions", 60)

        for subreddit in bounded_subreddits:
            retrieval_trace = {
                "subreddit": subreddit,
                "query": community_queries[subreddit],
                "requested_limit": posts_per_community,
            }
            raw_posts, retrieval_error, retrieval_elapsed = retrieval_by_subreddit[subreddit]
            if retrieval_error is not None:
                if trace is not None:
                    retrieval_trace.update({
                        "status": "error",
                        "error_type": type(retrieval_error).__name__,
                        "error": str(retrieval_error),
                        "retrieved_count": 0,
                        "usable_before_candidate_limit": [],
                        "ranker_input_ids": [],
                        "elapsed_seconds": retrieval_elapsed,
                    })
                    trace["retrieval"].append(retrieval_trace)
                logger.debug(
                    "Skipping r/%s after Reddit retrieval failure: %s",
                    subreddit,
                    retrieval_error,
                )
                continue

            raw_posts = raw_posts or []
            usable_before_candidate_limit = [
                post
                for post in raw_posts
                if _is_usable_post(post)
                and _belongs_to_subreddit(post, subreddit)
            ]
            usable = usable_before_candidate_limit[:posts_per_community]
            relevant_thread_leads = [
                post.model_copy(update={
                    "evidence_source": "arcticshift_retrieval",
                })
                for post in usable
                if _is_relevant_indexed_discussion(
                    post.title,
                    post.body,
                    description,
                )
            ][:3]
            if relevant_thread_leads:
                indexed_thread_leads.append((
                    f"reddit:{subreddit.casefold()}",
                    relevant_thread_leads,
                    {
                        "platform": "reddit",
                        "name": subreddit,
                        "url": f"https://www.reddit.com/r/{subreddit}/",
                        "kind": "community",
                    },
                ))

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
                    "elapsed_seconds": retrieval_elapsed,
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
        if progress_callback:
            progress_callback("ranking", 75)

        # ---------------------------------------------------------
        # 4. Add public indexed discussions, then rank all sources
        #    together so fallback/relevance tiers remain consistent.
        # ---------------------------------------------------------
        public_candidates = []
        public_metadata = {}
        ranked = []
        ranker_trace = {} if trace is not None else None
        ranker_started = None
        public_error = None
        if progress_callback:
            progress_callback("cross_checking", 85)
        try:
            public_candidates, public_metadata, public_status = public_discovery_future.result()
            if source_status is not None:
                source_status.update({
                    f"public_{platform}": status
                    for platform, status in public_status.items()
                })
            reddit_candidates = {
                candidate.subreddit.casefold(): candidate
                for candidate in candidates
            }
            for public_candidate in public_candidates:
                metadata = public_metadata.get(public_candidate.subreddit, {})
                lead_posts = [
                    post for post in public_candidate.posts
                    if post.evidence_source == "duckduckgo_index"
                ]
                fallback_posts = [
                    post for post in public_candidate.posts
                    if post.evidence_source in {
                        "duckduckgo_index",
                        "reddit_search_api",
                        "reddit_public_json",
                        "agent_reach_opencli",
                    }
                ]
                if fallback_posts:
                    indexed_thread_leads.append((
                        public_candidate.subreddit,
                        fallback_posts,
                        metadata,
                    ))
                if lead_posts:
                    public_candidate.posts = [
                        post for post in public_candidate.posts
                        if post.evidence_source != "duckduckgo_index"
                    ]
                if not public_candidate.posts:
                    continue
                if metadata.get("platform") == "reddit":
                    subreddit = public_candidate.subreddit.partition(":")[2]
                    if not subreddit:
                        continue
                    public_candidate.subreddit = subreddit
                    for post in public_candidate.posts:
                        post.subreddit = subreddit
                    existing = reddit_candidates.get(subreddit.casefold())
                    if existing is not None:
                        known_ids = {
                            post.id or post.url or post.title
                            for post in existing.posts
                        }
                        existing.posts.extend(
                            post
                            for post in public_candidate.posts
                            if (post.id or post.url or post.title) not in known_ids
                        )
                    else:
                        candidates.append(public_candidate)
                        reddit_candidates[subreddit.casefold()] = public_candidate
                else:
                    candidates.append(public_candidate)

            ranker_started = time.perf_counter() if trace is not None else None
            ranked = (
                self.ranker.rank(
                    description,
                    candidates,
                    limit=result_limit,
                    trace=ranker_trace,
                )
                if candidates
                else []
            )
            if not ranked:
                fallback_candidates = _discovery_evidence_candidates(
                    discovery_evidence,
                    queries,
                    MAX_DISCOVERY_FALLBACK_COMMUNITIES,
                )
                if fallback_candidates:
                    fallback_trace = {} if trace is not None else None
                    fallback_ranked = self.ranker.rank(
                        description,
                        fallback_candidates,
                        limit=result_limit,
                        trace=fallback_trace,
                    )
                    ranked = []
                    for result in fallback_ranked:
                        result["status"] = "potential"
                        result["score"] = round(
                            result.get("score", 0.0) * 0.6,
                            4,
                        )
                        result["evidence_source"] = "indexed_discovery"
                        ranked.append(result)
                    if source_status is not None:
                        source_status["reddit_discovery_fallback"] = (
                            "used" if ranked else "no_problem_evidence"
                        )
                    if trace is not None:
                        trace["discovery_fallback"] = {
                            "candidate_count": len(fallback_candidates),
                            "ranked_count": len(ranked),
                            "ranker": fallback_trace,
                        }
        except Exception as exc:
            public_error = exc
            ranked = []
            logger.warning("Public non-Reddit community discovery failed: %s", exc)
            if trace is not None:
                trace["public_discovery"] = {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            if source_status is not None:
                for platform in ("discord", "slack", "linkedin", "indiehackers"):
                    source_status[f"public_{platform}"] = "unavailable"
                for platform in ("reddit", "linkedin", "indiehackers"):
                    source_status[f"public_{platform}_discussions"] = "unavailable"

        # ---------------------------------------------------------
        # 4b. Always-on thread fallback: web-searched Reddit threads
        # ---------------------------------------------------------
        if len(ranked) < result_limit:
            try:
                web_leads = web_leads_future.result(timeout=25)
                indexed_thread_leads.extend(web_leads)
                if source_status is not None:
                    source_status["web_thread_fallback"] = (
                        "used" if web_leads else "no_threads_found"
                    )
            except Exception as exc:
                logger.warning("Web thread fallback failed: %s", exc)
                if source_status is not None:
                    source_status["web_thread_fallback"] = "unavailable"

        # ---------------------------------------------------------
        # 4c. Merge thread leads into the ranked results
        # ---------------------------------------------------------
        ranked_communities = {
            str(item.get("community_id") or item.get("subreddit") or "").casefold()
            for item in ranked
        }
        added_thread_leads = 0
        for subreddit, posts, metadata in indexed_thread_leads:
            community_id = (
                subreddit.partition(":")[2]
                if metadata.get("platform") == "reddit"
                else subreddit
            )
            if not community_id:
                continue
            identity = community_id.casefold()
            if identity in ranked_communities:
                existing_result = next(
                    (
                        item for item in ranked
                        if str(
                            item.get("community_id")
                            or item.get("subreddit")
                            or ""
                        ).casefold() == identity
                    ),
                    None,
                )
                if existing_result is not None:
                    known_urls = {
                        post.url for post in existing_result.get("top_posts", [])
                    }
                    existing_result.setdefault("top_posts", []).extend(
                        post for post in posts
                        if post.url not in known_urls
                    )
                continue
            if added_thread_leads >= result_limit:
                continue
            ranked_communities.add(identity)
            platform = metadata.get("platform", "reddit")
            ranked.append({
                "subreddit": community_id,
                "community_id": community_id,
                "name": metadata.get("name", community_id),
                "platform": platform,
                "url": metadata.get(
                    "url",
                    f"https://www.reddit.com/r/{community_id}/",
                ),
                "community_kind": metadata.get("kind", "community"),
                "score": 0.0,
                "status": "thread_lead",
                "evidence_source": "indexed_thread_lead",
                "signals": {
                    "problem_relevant_posts": 0,
                    "evidence_mean": 0.0,
                    "prevalence": 0.0,
                    "avg_recency": 0.0,
                },
                "analyzed_posts": 0,
                "top_posts": posts[:3],
            })
            added_thread_leads += 1

        if not any(
            item.get("status") in {"relevant", "potential", "suggestion"}
            for item in ranked
        ):
            # Always give the user a concrete place to start, while clearly
            # marking that no discussion evidence was verified.
            fallback_name = next(iter(community_queries), "Entrepreneur")
            ranked.append({
                "subreddit": fallback_name,
                "community_id": fallback_name,
                "name": fallback_name,
                "platform": "reddit",
                "url": f"https://www.reddit.com/r/{fallback_name}/",
                "community_kind": "community",
                "score": 0.0,
                "status": "suggestion",
                "evidence_source": "generic_fallback" if not community_queries else "discovery_candidate",
                "signals": {
                    "problem_relevant_posts": 0,
                    "evidence_mean": 0.0,
                    "prevalence": 0.0,
                    "avg_recency": 0.0,
                },
                "analyzed_posts": 0,
                "top_posts": [],
            })
            if source_status is not None:
                source_status["community_suggestion_fallback"] = (
                    "generic" if not community_queries else "discovery_candidate"
                )

        for result in ranked:
            key = result["subreddit"]
            metadata = public_metadata.get(key, {})
            if not metadata:
                metadata = public_metadata.get(
                    f"reddit:{key.casefold()}",
                    {},
                )
            result.update({
                "community_id": key,
                "name": metadata.get("name", result.get("name", key)),
                "platform": metadata.get("platform", result.get("platform", "reddit")),
                "url": metadata.get(
                    "url",
                    result.get("url") or f"https://www.reddit.com/r/{key}/",
                ),
                "community_kind": metadata.get(
                    "kind",
                    result.get("community_kind", "community"),
                ),
            })
            if not result.get("explanation"):
                result["explanation"] = _build_explanation(description, result)

        if trace is not None and public_error is None:
            trace["public_discovery"] = {
                "status": "success",
                "candidate_count": len(public_candidates),
                "ranked_communities": [
                    {
                        "platform": item.get("platform"),
                        "name": item.get("name"),
                        "score": item.get("score"),
                        "source_url": item.get("url"),
                    }
                    for item in ranked
                    if item.get("platform") != "reddit"
                ],
            }

        if trace is not None:
            trace["ranked_output"] = [
                {
                    "rank": index,
                    "subreddit": result.get("subreddit"),
                    "name": result.get("name"),
                    "platform": result.get("platform", "reddit"),
                    "url": result.get("url"),
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
            trace["community_ranker"] = ranker_trace
            if ranker_trace is not None and ranker_started is not None:
                trace["community_ranker"]["elapsed_seconds"] = time.perf_counter() - ranker_started

        # ---------------------------------------------------------
        # 5. Return only the requested number of communities
        # ---------------------------------------------------------
        selected = ranked[:result_limit]
        selected_object_ids = {id(result) for result in selected}
        selected.extend(
            result
            for result in ranked
            if result.get("status") == "thread_lead"
            and id(result) not in selected_object_ids
        )
        if trace is not None:
            trace["returned_output"] = [
                {
                    "rank": index,
                    "subreddit": result.get("subreddit"),
                    "name": result.get("name"),
                    "platform": result.get("platform", "reddit"),
                    "url": result.get("url"),
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
        if progress_callback:
            progress_callback("finishing", 96)
        return selected


def _serialize_post(post):
    if hasattr(post, "model_dump"):
        return post.model_dump(mode="json")
    if is_dataclass(post):
        return asdict(post)
    if hasattr(post, "__dict__"):
        return dict(vars(post))
    return {"value": str(post)}
