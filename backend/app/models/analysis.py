from pydantic import BaseModel
from app.models.reddit import RedditPost

class AnalyzeRequest(BaseModel):
    description: str

class CommunityResult(BaseModel):
    subreddit: str
    score: float
    posts: list[RedditPost]

class AnalyzeResponse(BaseModel):
    description: str
    communities: list[CommunityResult]