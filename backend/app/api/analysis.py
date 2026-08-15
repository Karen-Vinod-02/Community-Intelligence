from fastapi import APIRouter
from app.models.analysis import AnalyzeRequest, AnalyzeResponse, CommunityResult
from app.services.community import CommunityService

router = APIRouter()
community_service = CommunityService()

@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    ranked_communities = community_service.analyze(
        request.description,
        candidate_limit=10,
        result_limit=5,
        posts_per_community=5,
    )

    communities = [
        CommunityResult(
            subreddit=result["subreddit"],
            score=result["score"],
            posts=result["top_posts"],
        )
        for result in ranked_communities
    ]

    return AnalyzeResponse(
        description=request.description,
        communities=communities,
    )