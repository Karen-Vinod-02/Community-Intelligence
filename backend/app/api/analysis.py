from fastapi import APIRouter

from app.models.analysis import AnalyzeRequest, AnalyzeResponse
from app.services.reddit.service import RedditService

router = APIRouter()
reddit_service = RedditService()

@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    posts = reddit_service.search_posts(
        subreddit="startups",
        limit=5,
    )

    return AnalyzeResponse(
        description=request.description,
        posts=posts,
    )