import json
import logging
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import quote_plus, urlparse

import requests

from app.models.reddit import RedditPost
from app.services.ranking import CommunityCandidate
from app.services.query_extraction import (
    PRODUCT_WRAPPER_WORDS,
    extract_problem_hypotheses,
)
from app.services.web_search import search_text
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


logger = logging.getLogger(__name__)
MAX_RESULTS_PER_SEARCH = 8
MAX_DISCUSSION_RESULTS_PER_SEARCH = 8
MAX_PUBLIC_CANDIDATES = 24
MIN_DISCUSSION_TERM_OVERLAP = 2
_PLACEHOLDER_SEARCH_TEXT = {
    "",
    "link to reddit.com",
    "the site owner hides the web page description.",
    "the site owner hides the web page description",
}
_BUDGET_DISCUSSION_CONTEXT = re.compile(
    r"\b(?:budgeting|expenses?|spending|saving|savings|allowance)\b"
    r"|\b(?:i|we|you|they)\s+(?:actually\s+)?budget\b"
    r"|\bstudents?\s+(?:are\s+)?budgeting\b"
    r"|\b(?:budget|money)\s+(?:app|tool|plan|tips|tracker|tracking)\b"
    r"|\b(?:monthly|weekly|semester)\s+budget\b"
    r"|\bbudget\s+for\s+(?:a\s+)?(?:student|college)\b"
    r"|\b(?:track|manage|plan|save|spend)\w*\b.{0,40}"
    r"\b(?:budget|money|expenses?)\b",
    re.IGNORECASE,
)

PLATFORMS = {
    "discord": {
        "query": "site:discord.com/servers OR site:disboard.org/server OR site:discordservers.com/server",
        "pattern": re.compile(
            r"(?:discord\.com/servers|disboard\.org/server|discordservers\.com/server)/([^/?#]+)",
            re.IGNORECASE,
        ),
    },
    "slack": {
        "query": "site:slack.com/community OR site:slack.com/communities OR site:join.slack.com OR site:slackcommunity.com",
        "pattern": re.compile(
            r"(?:slack\.com/(?:communities|community)|join\.slack\.com|slackcommunity\.com)/([^?#]+)",
            re.IGNORECASE,
        ),
    },
    "linkedin": {
        "query": "site:linkedin.com/groups",
        "pattern": re.compile(r"linkedin\.com/groups/([^/?#]+)", re.IGNORECASE),
    },
    "indiehackers": {
        "query": "site:indiehackers.com/group OR site:indiehackers.com/groups",
        "pattern": re.compile(
            r"indiehackers\.com/groups?/([^/?#]+)",
            re.IGNORECASE,
        ),
    },
}

DISCUSSION_SOURCES = {
    "reddit": {
        "query": "site:reddit.com/r/",
        "pattern": re.compile(
            r"reddit\.com/r/([A-Za-z0-9_]+)/comments/([A-Za-z0-9]+)/",
            re.IGNORECASE,
        ),
    },
    "linkedin": {
        "query": "site:linkedin.com/posts/ OR site:linkedin.com/feed/update/",
        "pattern": re.compile(
            r"linkedin\.com/(?:posts/|feed/update/)",
            re.IGNORECASE,
        ),
    },
    "indiehackers": {
        "query": "site:indiehackers.com/post/",
        "pattern": re.compile(
            r"indiehackers\.com/post/([^/?#]+)",
            re.IGNORECASE,
        ),
    },
    "hackernews": {
        "query": "",
        "pattern": re.compile(
            r"news\.ycombinator\.com/item\?id=(\d+)",
            re.IGNORECASE,
        ),
    },
}


def _normalise_discussion_term(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _search(
    platform: str,
    query: str,
) -> tuple[str, list[dict], Exception | None]:
    search_query = f'{PLATFORMS[platform]["query"]} "{query}"'
    results, _, error = search_text(
        search_query,
        max_results=MAX_RESULTS_PER_SEARCH,
        timelimit="m",
    )
    if error is not None:
        logger.warning(
            "Public %s discovery failed on all search backends: %s",
            platform,
            error,
        )
    return platform, results, error


def _search_discussions(
    platform: str,
    query: str,
) -> tuple[str, str, list[dict], Exception | None]:
    if platform == "hackernews":
        try:
            response = requests.get(
                "https://hn.algolia.com/api/v1/search_by_date",
                params={
                    "query": query,
                    "tags": "story",
                    "hitsPerPage": MAX_DISCUSSION_RESULTS_PER_SEARCH,
                },
                timeout=(2, 5),
            )
            response.raise_for_status()
            payload = response.json()
            now = datetime.now(timezone.utc).timestamp()
            results = []
            for hit in payload.get("hits", []):
                if not isinstance(hit, dict):
                    continue
                created = hit.get("created_at_i")
                if not isinstance(created, (int, float)):
                    continue
                age_seconds = now - created
                if age_seconds < -86400 or age_seconds > 90 * 86400:
                    continue
                object_id = str(hit.get("objectID") or "").strip()
                title = str(hit.get("title") or "").strip()
                if not object_id or not title:
                    continue
                results.append({
                    "href": f"https://news.ycombinator.com/item?id={object_id}",
                    "title": title,
                    "body": str(hit.get("story_text") or "").strip(),
                    "date": created,
                })
            return platform, query, results, None
        except (requests.RequestException, ValueError, TypeError) as exc:
            logger.warning("Hacker News discussion search failed: %s", exc)
            return platform, query, [], exc

    site_query = DISCUSSION_SOURCES[platform]["query"]
    search_queries = [f'{site_query} "{query}"']
    if platform == "reddit":
        search_queries.append(f"{site_query} {query}")
    results = []
    error = None
    seen_urls = set()
    for search_query in search_queries:
        found, _, search_error = search_text(
            search_query,
            max_results=MAX_DISCUSSION_RESULTS_PER_SEARCH,
            timelimit="m",
            preferred_backend="duckduckgo",
        )
        if search_error is not None:
            error = search_error
        for item in found:
            url = item.get("href") or item.get("url") or ""
            if url and url not in seen_urls:
                seen_urls.add(url)
                results.append(item)
        if len(results) >= MAX_DISCUSSION_RESULTS_PER_SEARCH:
            break
    if platform == "reddit" and not results:
        results = _search_reddit_api(query)
        if not results:
            results = _search_agent_reach_reddit(query)
    if results:
        error = None
    if error is not None:
        logger.warning("Indexed %s discussion search failed: %s", platform, error)
    return platform, query, results, error


def _timestamp(result: dict) -> float | None:
    value = result.get("date") or result.get("published") or result.get("timestamp")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except ValueError:
            return None
    return None


def _discussion_terms(description: str) -> set[str]:
    hypotheses = extract_problem_hypotheses(description)
    if hypotheses:
        text = " ".join(
            " ".join((
                hypothesis.actor or "",
                hypothesis.action or "",
                hypothesis.target or "",
                hypothesis.activity or "",
                hypothesis.interaction or "",
            ))
            for hypothesis in hypotheses
        )
    else:
        text = description
    return {
        _normalise_discussion_term(token)
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) > 2
        and token not in ENGLISH_STOP_WORDS
        and token not in PRODUCT_WRAPPER_WORDS
    }


def _is_relevant_indexed_discussion(
    title: str,
    snippet: str,
    description: str,
) -> bool:
    if (
        title.casefold().strip() in _PLACEHOLDER_SEARCH_TEXT
        or snippet.casefold().strip() in _PLACEHOLDER_SEARCH_TEXT
    ):
        return False
    query_terms = _discussion_terms(description)
    result_terms = {
        _normalise_discussion_term(token)
        for token in re.findall(r"[a-z0-9]+", f"{title} {snippet}".casefold())
    }
    combined_text = f"{title} {snippet}"
    if len(query_terms & result_terms) < MIN_DISCUSSION_TERM_OVERLAP:
        social_intent = bool(re.search(
            r"\b(?:partners?|budd(?:y|ies)|study groups?)\b",
            description,
            re.IGNORECASE,
        ))
        study_together = bool(re.search(
            r"\b(?:study|studying)\b.{0,60}\b(?:together|group|partner)\b"
            r"|\b(?:together|group|partner)\b.{0,60}\b(?:study|studying)\b",
            combined_text,
            re.IGNORECASE,
        ))
        active_seeking = bool(re.search(
            r"\b(?:looking for|need|seeking|want|forming|join)\b",
            combined_text,
            re.IGNORECASE,
        ))
        if not (social_intent and study_together and active_seeking):
            return False
    hypotheses = extract_problem_hypotheses(description)
    if any(hypothesis.action == "budget" for hypothesis in hypotheses):
        return bool(_BUDGET_DISCUSSION_CONTEXT.search(combined_text))
    return True


def _verified_reddit_discussion(
    result: dict,
    recent_days: int,
) -> tuple[str, RedditPost] | None:
    url = str(result.get("href") or result.get("url") or "").split("?", 1)[0]
    match = DISCUSSION_SOURCES["reddit"]["pattern"].search(url)
    if not match:
        return None

    try:
        response = requests.get(
            f"{url.rstrip('/')}.json",
            params={"raw_json": 1},
            headers={"User-Agent": "CommInt/0.1 (community research prototype)"},
            timeout=(1.5, 3),
        )
        response.raise_for_status()
        payload = response.json()
        post_data = payload[0]["data"]["children"][0]["data"]
        created = float(post_data["created_utc"])
        age_seconds = datetime.now(timezone.utc).timestamp() - created
        if age_seconds < -86400 or age_seconds > recent_days * 86400:
            return None

        title = str(post_data.get("title") or "").strip()
        body = str(post_data.get("selftext") or "").strip()
        if (
            not title
            or title.casefold() in {"[removed]", "[deleted]"}
            or body.casefold() in {"[removed]", "[deleted]"}
        ):
            return None

        subreddit = str(post_data.get("subreddit") or match.group(1))
        post_url = f"https://www.reddit.com{post_data.get('permalink') or urlparse(url).path}"
        return subreddit, RedditPost(
            id=str(post_data.get("id") or match.group(2)),
            subreddit=subreddit,
            title=title[:500],
            body=body[:1200],
            score=post_data.get("score"),
            num_comments=post_data.get("num_comments"),
            created_utc=created,
            url=post_url,
            recency_verified=True,
            evidence_source="reddit_public_json",
        )
    except (
        requests.RequestException,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
    ) as exc:
        logger.debug("Could not verify indexed Reddit discussion %s: %s", url, exc)
        return None


def _search_reddit_api(query: str) -> list[dict]:
    try:
        response = requests.get(
            "https://www.reddit.com/search.json",
            params={
                "q": query,
                "sort": "relevance",
                "t": "month",
                "type": "link",
                "limit": MAX_DISCUSSION_RESULTS_PER_SEARCH,
                "raw_json": 1,
            },
            headers={"User-Agent": "CommInt/0.1 (community research prototype)"},
            timeout=(1.5, 3),
        )
        response.raise_for_status()
        payload = response.json()
        now = datetime.now(timezone.utc).timestamp()
        results = []
        for child in payload.get("data", {}).get("children", []):
            post_data = child.get("data", {})
            created = float(post_data.get("created_utc") or 0)
            age = now - created
            title = str(post_data.get("title") or "").strip()
            subreddit = str(post_data.get("subreddit") or "").strip()
            permalink = str(post_data.get("permalink") or "").strip()
            if (
                not title
                or not subreddit
                or not permalink
                or age < -86400
                or age > 90 * 86400
            ):
                continue
            results.append({
                "href": f"https://www.reddit.com{permalink}",
                "title": title,
                "body": str(post_data.get("selftext") or "").strip(),
                "date": created,
                "id": str(post_data.get("id") or ""),
                "subreddit": subreddit,
                "score": post_data.get("score"),
                "num_comments": post_data.get("num_comments"),
                "verified_source": "reddit_json",
            })
        return results
    except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
        logger.info("Reddit public search fallback failed: %s", exc)
        return []


def _search_agent_reach_reddit(query: str) -> list[dict]:
    executable = shutil.which("opencli")
    if not executable:
        return []
    arguments = [
        executable,
        "reddit",
        "search",
        query,
        "--time",
        "month",
        "--sort",
        "relevance",
        "--limit",
        str(MAX_DISCUSSION_RESULTS_PER_SEARCH),
        "--window",
        "background",
        "--site-session",
        "ephemeral",
        "-f",
        "json",
    ]
    if executable.lower().endswith(".ps1"):
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if not powershell:
            return []
        arguments = [powershell, "-NoProfile", "-File", *arguments]
    try:
        completed = subprocess.run(
            arguments,
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        if completed.returncode != 0:
            logger.info(
                "Agent Reach Reddit search returned exit code %d: %s",
                completed.returncode,
                completed.stderr[-500:],
            )
            return []
        payload, _ = json.JSONDecoder().raw_decode(completed.stdout.lstrip())
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        results = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            url = str(row.get("url") or "").strip()
            subreddit = str(row.get("subreddit") or "").removeprefix("r/").strip()
            title = str(row.get("title") or "").strip()
            created = row.get("created_utc")
            if not url or not subreddit or not title or not created:
                continue
            results.append({
                "href": url,
                "title": title,
                "body": str(row.get("selftext") or "").strip(),
                "date": float(created),
                "id": str(row.get("id") or ""),
                "subreddit": subreddit,
                "score": row.get("score"),
                "num_comments": row.get("comments"),
                "verified_source": "agent_reach_opencli",
            })
        return results
    except (OSError, subprocess.SubprocessError, ValueError, TypeError) as exc:
        logger.info("Agent Reach Reddit search fallback failed: %s", exc)
        return []


def _name_from_result(platform: str, slug: str, title: str) -> str:
    cleaned_title = re.split(r"\s[|–—-]\s", title, maxsplit=1)[0].strip()
    if cleaned_title and len(cleaned_title) > 3:
        return cleaned_title[:100]
    return slug.replace("-", " ").replace("_", " ").strip()[:100]


def discover_public_communities(
    description: str,
    queries: list[str],
    recent_days: int = 90,
) -> tuple[list[CommunityCandidate], dict[str, dict]]:
    """Find public community pages and recent indexed discussion threads."""
    search_queries = list(dict.fromkeys([*queries[:2], description]))
    jobs = [
        (platform, query)
        for platform in PLATFORMS
        for query in search_queries
    ]
    discussion_jobs = [
        (platform, query)
        for platform in DISCUSSION_SOURCES
        for query in search_queries
    ]
    with ThreadPoolExecutor(
        max_workers=min(4, len(jobs) + len(discussion_jobs))
    ) as executor:
        community_future = executor.map(lambda job: _search(*job), jobs)
        discussion_future = executor.map(
            lambda job: _search_discussions(*job),
            discussion_jobs,
        )
        search_results = list(community_future)
        discussion_results = list(discussion_future)

    source_status = {}
    for platform in PLATFORMS:
        attempts = [
            error
            for result_platform, _, error in search_results
            if result_platform == platform
        ]
        failures = sum(error is not None for error in attempts)
        source_status[platform] = (
            "unavailable"
            if attempts and failures == len(attempts)
            else "partial_failure"
            if failures
            else "available"
        )
    for platform in DISCUSSION_SOURCES:
        attempts = [
            error
            for result_platform, _, _, error in discussion_results
            if result_platform == platform
        ]
        failures = sum(error is not None for error in attempts)
        source_status[f"{platform}_discussions"] = (
            "unavailable"
            if attempts and failures == len(attempts)
            else "partial_failure"
            if failures
            else "available"
        )

    communities: dict[str, dict] = {}
    for platform, result_set, error in search_results:
        if error is not None:
            continue
        pattern = PLATFORMS[platform]["pattern"]
        for result in result_set:
            url = result.get("href") or result.get("url") or ""
            match = pattern.search(url)
            if not match:
                continue
            slug = match.group(1).strip("/")
            title = (result.get("title") or "").strip()
            snippet = " ".join(str(result.get(key) or "").strip() for key in (
                "body", "snippet", "description",
            )).strip()
            if not slug or not (title or snippet):
                continue
            key = f"{platform}:{slug.casefold()}"
            entry = communities.setdefault(key, {
                "platform": platform,
                "name": _name_from_result(platform, slug, title),
                "url": url.split("?")[0],
                "posts": [],
            })

            created = _timestamp(result)
            if created is not None:
                age_seconds = datetime.now(timezone.utc).timestamp() - created
                if age_seconds > recent_days * 86400 or age_seconds < -86400:
                    continue

            if not title and not snippet:
                continue
            entry["posts"].append(RedditPost(
                id=url,
                subreddit=key,
                title=title or entry["name"],
                body=snippet[:1500],
                created_utc=created,
                url=url,
                recency_verified=created is not None,
            ))

    seen_discussion_urls = set()
    for platform, query, result_set, error in discussion_results:
        if error is not None:
            continue

        pattern = DISCUSSION_SOURCES[platform]["pattern"]
        for result in result_set:
            url = result.get("href") or result.get("url") or ""
            match = pattern.search(url)
            if not match:
                continue

            canonical_url = url.split("#", 1)[0]
            if canonical_url in seen_discussion_urls:
                continue

            reddit_post = None
            indexed_lead = False
            if platform == "reddit":
                verified_source = result.get("verified_source")
                if verified_source in {"reddit_json", "agent_reach_opencli"}:
                    subreddit_name = str(result.get("subreddit") or match.group(1))
                    title = str(result.get("title") or "").strip()
                    snippet = str(result.get("body") or "").strip()
                    if not _is_relevant_indexed_discussion(
                        title,
                        snippet,
                        description,
                    ):
                        continue
                    post_data_url = urlparse(url)
                    reddit_post = RedditPost(
                        id=str(result.get("id") or match.group(2)),
                        subreddit=subreddit_name,
                        title=str(result.get("title") or "").strip()[:500],
                        body=str(result.get("body") or "").strip()[:1200],
                        score=result.get("score"),
                        num_comments=result.get("num_comments"),
                        created_utc=_timestamp(result),
                        url=f"{post_data_url.scheme}://{post_data_url.netloc}{post_data_url.path}",
                        recency_verified=True,
                        evidence_source=(
                            "agent_reach_opencli"
                            if verified_source == "agent_reach_opencli"
                            else "reddit_search_api"
                        ),
                    )
                    subreddit_name = reddit_post.subreddit
                else:
                    verified_discussion = _verified_reddit_discussion(
                        result,
                        recent_days,
                    )
                    if verified_discussion is None:
                        subreddit_name = match.group(1)
                        indexed_lead = True
                    else:
                        subreddit_name, reddit_post = verified_discussion
                        canonical_url = reddit_post.url or canonical_url
                if not reddit_post and not indexed_lead:
                    subreddit_name = match.group(1)

            title = (result.get("title") or "").strip()
            snippet = " ".join(
                str(result.get(key) or "").strip()
                for key in ("body", "snippet", "description")
            ).strip()
            if reddit_post is not None:
                title = reddit_post.title
                snippet = reddit_post.body
                if not _is_relevant_indexed_discussion(
                    title,
                    snippet,
                    description,
                ):
                    continue
            if indexed_lead and not _is_relevant_indexed_discussion(
                title,
                snippet,
                description,
            ):
                continue
            if not title or (
                not snippet
                and platform not in {"hackernews", "reddit"}
            ):
                continue

            created = (
                reddit_post.created_utc
                if reddit_post is not None
                else None
                if indexed_lead
                else _timestamp(result)
            )
            now = datetime.now(timezone.utc).timestamp()
            if created is None:
                if not indexed_lead:
                    continue
                if not _is_relevant_indexed_discussion(
                    title,
                    snippet,
                    description,
                ):
                    continue
                indexed_lead = True
            elif (
                now - created > recent_days * 86400
                or created - now > 86400
            ):
                continue

            if platform == "reddit":
                community_name = subreddit_name
                key = f"reddit:{subreddit_name.casefold()}"
                community_url = f"https://www.reddit.com/r/{subreddit_name}/"
            elif platform == "indiehackers":
                community_name = "Indie Hackers public discussions"
                key = "indiehackers:public-discussions"
                community_url = "https://www.indiehackers.com/"
            elif platform == "hackernews":
                community_name = "Hacker News public discussions"
                key = "hackernews:public-discussions"
                community_url = "https://news.ycombinator.com/"
            else:
                community_name = "LinkedIn public discussions"
                key = "linkedin:public-discussions"
                community_url = (
                    "https://www.linkedin.com/search/results/content/"
                    f"?keywords={quote_plus(query)}"
                )

            seen_discussion_urls.add(canonical_url)
            entry = communities.setdefault(key, {
                "platform": platform,
                "name": community_name,
                "url": community_url,
                "kind": (
                    "community"
                    if platform == "reddit"
                    else "indexed_discussions"
                ),
                "posts": [],
            })
            entry["posts"].append(
                reddit_post
                if reddit_post is not None
                else RedditPost(
                    id=canonical_url,
                    subreddit=key,
                    title=title[:500],
                    body=snippet[:1200],
                    created_utc=created,
                    url=canonical_url,
                    recency_verified=created is not None,
                    evidence_source=(
                        "duckduckgo_index"
                        if indexed_lead
                        else None
                    ),
                )
            )

    candidates = []
    metadata = {}
    for key, entry in list(communities.items())[:MAX_PUBLIC_CANDIDATES]:
        posts = entry["posts"]
        if not posts:
            continue
        candidates.append(CommunityCandidate(
            subreddit=key,
            posts=posts[:5],
        ))
        metadata[key] = {
            "platform": entry["platform"],
            "name": entry["name"],
            "url": entry["url"],
            "kind": entry.get("kind", "community"),
        }

    return candidates, metadata, source_status
