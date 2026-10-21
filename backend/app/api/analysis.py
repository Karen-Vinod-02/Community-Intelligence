import logging
import re

import requests
from fastapi import APIRouter, Header, HTTPException
from app.models.analysis import (
    AnalyzeRequest,
    AnalyzeResponse,
    CompetitorsResponse,
    CommunityExplanation,
    CommunityResult,
    DiscussionMessageRequest,
    DiscussionMessageResponse,
    EvidenceMatch,
)
from app.services.community import CommunityService, _belongs_to_subreddit
from app.services.evaluation_trace import (
    EvaluationTraceNotConfigured,
    evaluation_trace_directory,
    write_evaluation_trace,
)
from app.services.competitors import find_competitors
from app.services.generation import GenerationUnavailable, generate_outreach
from app.services.analysis_cache import analysis_cache
from app.services.analysis_progress import (
    get_analysis_progress,
    update_analysis_progress,
)


router = APIRouter()
community_service = CommunityService()
logger = logging.getLogger(__name__)

FRAME_LABELS = {
    "actor_alignment": "Target user",
    "action_alignment": "User action",
    "target_alignment": "Desired outcome",
    "activity_alignment": "Activity",
    "interaction_alignment": "Interaction",
    "friction_alignment": "Problem friction",
    "goal_alignment": "User goal",
}

def _build_explanation(result: dict) -> CommunityExplanation:
    if result.get("status") == "suggestion":
        return CommunityExplanation(
            summary=(
                "Starting-point suggestion only. No recent discussion was verified "
                "for this community, so treat it as a place to explore rather than a match."
            ),
            matches=[],
        )
    signals = result.get("signals") or {}
    relevant_count = signals.get("problem_relevant_posts", 0)
    analyzed_count = result.get("analyzed_posts", 0)
    if result.get("platform", "reddit") == "reddit":
        top_post = next(iter(result.get("top_posts", [])), None)
        evidence_title = getattr(top_post, "title", "").strip()
        evidence_body = getattr(top_post, "body", "").strip()
        evidence = evidence_title or evidence_body[:220].strip()
        if evidence_title and evidence_body:
            evidence += f" — {evidence_body[:220].strip()}"
        if result.get("evidence_source") == "indexed_discovery":
            verification_note = (
                "Reddit's public post data confirms the discussion and its recent "
                "date; overall community activity is not independently verified."
                if top_post
                and getattr(top_post, "recency_verified", False)
                and getattr(top_post, "created_utc", None) is not None
                else "Post recency and community activity could not be verified."
            )
            summary = (
                f"Fallback match from indexed search evidence: “{evidence}”. "
                f"{verification_note}"
                if evidence
                else (
                    "Fallback match from indexed search evidence. "
                    f"{verification_note}"
                )
            )
        else:
            summary = (
                f"Strongest evidence: “{evidence}”. "
                f"{relevant_count} of {analyzed_count} recent posts cleared the "
                "problem-evidence gate."
                if evidence
                else (
                    f"{relevant_count} of {analyzed_count} recent posts cleared the "
                    "problem-evidence gate."
                )
            )
    else:
        summary = (
            f"{relevant_count or len(result.get('top_posts', []))} recent discussion(s) "
            f"in {result.get('name') or 'this community'} passed the problem relevance check."
        )

    matches = []
    for post in result.get("top_posts", [])[:3]:
        frame_scores = post.problem_match or {}
        if frame_scores:
            key, score = max(frame_scores.items(), key=lambda item: item[1])
            label = FRAME_LABELS.get(key, "Problem match")
        else:
            label = "Problem evidence"
            score = result.get("evidence_score", result.get("score", 0.0))

        text = post.title or post.body[:240]
        if post.body and post.title:
            text = f"{post.title} — {post.body[:180].strip()}"
        matches.append(EvidenceMatch(
            label=label,
            score=score,
            text=text,
            source_url=post.url,
        ))

    return CommunityExplanation(summary=summary, matches=matches)


def _fetch_community_rules(name: str) -> tuple[list[str], bool]:
    if not re.fullmatch(r"[A-Za-z0-9_]{2,21}", name):
        return [], False
    try:
        response = requests.get(
            f"https://www.reddit.com/r/{name}/about/rules.json",
            headers={"User-Agent": "CommInt/0.1 (community research prototype)"},
            timeout=(1.5, 4),
        )
        response.raise_for_status()
        payload = response.json()
        rules = payload.get("rules", [])
        return [
            str(rule.get("short_name") or rule.get("description") or "").strip()
            for rule in rules
            if isinstance(rule, dict)
            and (rule.get("short_name") or rule.get("description"))
        ], True
    except (requests.RequestException, ValueError, AttributeError) as exc:
        logger.info("Could not verify public rules for r/%s: %s", name, exc)
        return [], False


@router.get("/analysis-progress/{progress_id}")
def analysis_progress(progress_id: str):
    return get_analysis_progress(progress_id)


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    request: AnalyzeRequest,
    progress_id: str | None = Header(default=None, alias="X-Analysis-Progress-ID"),
):
    update_analysis_progress(progress_id, "starting", 2)
    trace_directory = None
    if request.capture_trace:
        try:
            trace_directory = evaluation_trace_directory()
        except EvaluationTraceNotConfigured as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    trace = {} if request.capture_trace else None
    source_status = {}
    cache_key = analysis_cache.key(request.description)
    if trace is None:
        cached = analysis_cache.get(cache_key)
        cached_communities = cached.get("communities", []) if cached else []
        if cached_communities and all(
            item.get("status") == "relevant"
            and any(
                post.get("evidence_status") == "problem_relevant"
                and post.get("recency_verified")
                for post in item.get("posts", [])
            )
            for item in cached_communities
        ):
            update_analysis_progress(progress_id, "complete", 100)
            return AnalyzeResponse.model_validate({
                **cached,
                "description": request.description,
            })

    try:
        ranked_communities = community_service.analyze(
            request.description,
            candidate_limit=10,
            result_limit=5,
            posts_per_community=10,
            trace=trace,
            source_status=source_status,
            progress_callback=lambda stage, percent: update_analysis_progress(
                progress_id,
                stage,
                percent,
            ),
        )
    except Exception as exc:
        update_analysis_progress(progress_id, "failed", 100)
        if trace is not None:
            trace["pipeline_error"] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            write_evaluation_trace(trace, directory=trace_directory)
        raise

    communities = []
    for result in ranked_communities:
        platform = result.get("platform", "reddit")
        if result.get("status") not in {"relevant", "potential", "suggestion"}:
            continue
        community_name = result.get("name") or result.get("subreddit")
        community_url = result.get("url") or (
            f"https://www.reddit.com/r/{community_name}/"
        )
        posts = result.get("top_posts", [])
        if platform == "reddit":
            posts = [
                post
                for post in posts
                if _belongs_to_subreddit(
                    post,
                    result.get("community_id") or community_name,
                )
            ]
        posts = [
            post for post in posts
            if getattr(post, "evidence_status", None) == "problem_relevant"
            and getattr(post, "recency_verified", False)
            and getattr(post, "created_utc", None) is not None
        ]
        if not posts and result.get("status") != "suggestion":
            continue
        communities.append(CommunityResult(
            subreddit=community_name if platform == "reddit" else None,
            name=community_name,
            platform=platform,
            community_kind=result.get("community_kind", "community"),
            url=community_url,
            score=result["score"],
            posts=posts,
            analyzed_posts=result.get("analyzed_posts", 0),
            status=result.get("status", "relevant"),
            rank_tier=result.get("status", "relevant"),
            signals=result.get("signals"),
            explanation=_build_explanation(result),
        ))

    response = AnalyzeResponse(
        description=request.description,
        communities=communities,
        sources=source_status,
            fallback_used=any(item.status != "relevant" for item in communities),
    )
    update_analysis_progress(progress_id, "complete", 100)

    if request.capture_trace:
        trace["final_response"] = response.model_dump(mode="json")
        try:
            trace_id = write_evaluation_trace(trace, directory=trace_directory)
        except EvaluationTraceNotConfigured as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        response.evaluation_trace_id = trace_id
    else:
        cached_response = response.model_dump(mode="json")
        cached_response.pop("description", None)
        cached_response.pop("evaluation_trace_id", None)
        if communities:
            analysis_cache.set(cache_key, cached_response)

    return response


@router.post("/outreach", response_model=DiscussionMessageResponse)
def generate_community_outreach(request: DiscussionMessageRequest):
    if not request.evidence_posts and not request.evidence_titles:
        raise HTTPException(
            status_code=422,
            detail="A recent discussion is required to draft a specific contribution.",
        )

    rules = list(request.community_rules)
    rules_verified = False
    if request.platform.casefold() == "reddit":
        name = request.community_name
        if name.startswith("r/"):
            name = name[2:]
        fetched_rules, rules_verified = _fetch_community_rules(name)
        rules = fetched_rules or rules

    try:
        message, provider, model = generate_outreach(
            description=request.description,
            platform=request.platform,
            community_name=request.community_name,
            community_url=request.community_url or "",
            rules=rules,
            evidence_titles=request.evidence_titles,
            evidence_posts=[
                post.model_dump()
                for post in request.evidence_posts[:3]
            ],
        )
    except GenerationUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except requests.RequestException as exc:
        logger.exception("Outreach generation provider failed")
        raise HTTPException(
            status_code=502,
            detail="Outreach generation failed at the configured model provider.",
        ) from exc

    return DiscussionMessageResponse(
        message=message,
        provider=provider,
        model=model,
        rules_verified=rules_verified,
        review_notice="Give it a quick edit so it sounds like you before posting.",
    )


@router.post("/competitors", response_model=CompetitorsResponse)
def discover_competitors(request: AnalyzeRequest):
    try:
        competitors = find_competitors(request.description)
    except Exception as exc:
        logger.warning("Competitor discovery unavailable: %s", exc, exc_info=True)
        return CompetitorsResponse(
            competitors=[],
            status="search_unavailable",
        )

    return CompetitorsResponse(
        competitors=competitors,
        status=(
            "evidence_found"
            if competitors
            else "no_evidence_found"
        ),
    )
