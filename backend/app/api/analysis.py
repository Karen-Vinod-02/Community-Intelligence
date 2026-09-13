from fastapi import APIRouter
from app.models.analysis import AnalyzeRequest, AnalyzeResponse, CommunityResult
from app.services.community import CommunityService, _belongs_to_subreddit

router = APIRouter()
community_service = CommunityService()

@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    ranked_communities = community_service.analyze(
        request.description,
        candidate_limit=10,
        result_limit=5,
        posts_per_community=15,
    )

    communities = [
        CommunityResult(
            subreddit=result["subreddit"],
            score=result["score"],
            posts=[
                post
                for post in result["top_posts"]
                if _belongs_to_subreddit(post, result["subreddit"])
            ],
        )
        for result in ranked_communities
        if result["signals"].get("problem_relevant_posts", 0) > 0
    ]

    return AnalyzeResponse(
        description=request.description,
        communities=communities,
    )