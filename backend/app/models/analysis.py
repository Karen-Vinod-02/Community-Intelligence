from pydantic import BaseModel
from app.models.reddit import RedditPost

class AnalyzeRequest(BaseModel):
    description: str

class AnalyzeResponse(BaseModel):
    description: str
    posts: list[RedditPost]