from pydantic import BaseModel
from app.models.reddit import RedditPost


class AnalyzeRequest(BaseModel):
    description: str
    capture_trace: bool = False


class CommunitySignals(BaseModel):
    problem_relevant_posts: int
    evidence_mean: float
    prevalence: float
    avg_recency: float


class CommunityResult(BaseModel):
    subreddit: str
    score: float
    posts: list[RedditPost]

    # "relevant"  -- qualified via the strong problem_relevant gate.
    # "potential" -- fallback tier, only ever present when NO
    #                candidate in the response reached "relevant".
    #                Surface this to the frontend as a visibly
    #                different confidence level, not as equal-quality
    #                output.
    status: str = "relevant"

    # Exposed for explainability: why the community was surfaced.
    signals: CommunitySignals | None = None


class AnalyzeResponse(BaseModel):
    description: str
    communities: list[CommunityResult]

    # True when none of the returned communities reached the
    # "relevant" tier and the response is showing best-available
    # "potential" candidates instead. Lets the frontend show a
    # "these are lower-confidence matches" banner instead of silently
    # presenting potential-tier results as equally strong.
    fallback_used: bool = False

    # Present only when this request opted into server-side evaluation tracing.
    evaluation_trace_id: str | None = None
