from fastapi import APIRouter, HTTPException
from app.models.analysis import AnalyzeRequest, AnalyzeResponse, CommunityResult
from app.services.community import CommunityService, _belongs_to_subreddit
from app.services.evaluation_trace import (
    EvaluationTraceNotConfigured,
    evaluation_trace_directory,
    write_evaluation_trace,
)

router = APIRouter()
community_service = CommunityService()


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    trace_directory = None
    if request.capture_trace:
        try:
            trace_directory = evaluation_trace_directory()
        except EvaluationTraceNotConfigured as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    trace = {} if request.capture_trace else None
    try:
        ranked_communities = community_service.analyze(
            request.description,
            candidate_limit=10,
            result_limit=5,
            posts_per_community=15,
            trace=trace,
        )
    except Exception as exc:
        if trace is not None:
            trace["pipeline_error"] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            write_evaluation_trace(trace, directory=trace_directory)
        raise

    communities = [
        CommunityResult(
            subreddit=result["subreddit"],
            score=result["score"],
            posts=[
                post
                for post in result["top_posts"]
                if _belongs_to_subreddit(post, result["subreddit"])
            ],
            status=result.get("status", "relevant"),
            signals=result.get("signals"),
        )
        for result in ranked_communities
        if (result.get("signals") or {}).get("problem_relevant_posts", 0) > 0
    ]

    response = AnalyzeResponse(
        description=request.description,
        communities=communities,
        fallback_used=bool(communities) and all(
            community.status == "potential"
            for community in communities
        ),
    )

    if request.capture_trace:
        trace["final_response"] = response.model_dump(mode="json")
        try:
            trace_id = write_evaluation_trace(trace, directory=trace_directory)
        except EvaluationTraceNotConfigured as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        response.evaluation_trace_id = trace_id

    return response
