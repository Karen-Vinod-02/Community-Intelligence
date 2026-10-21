from pydantic import BaseModel


class RedditPost(BaseModel):
    id: str | None = None
    subreddit: str
    title: str
    body: str = ""
    author: str | None = None
    score: int | None = None
    num_comments: int | None = None
    created_utc: float | None = None
    url: str | None = None
    evidence_status: str | None = None
    evidence_source: str | None = None
    problem_match: dict[str, float] | None = None
    recency_verified: bool = False


class RedditComment(BaseModel):
    id: str | None = None
    body: str
    author: str | None = None
    score: int | None = None