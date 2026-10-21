from pydantic import BaseModel, Field
from app.models.reddit import RedditPost


class EvidenceMatch(BaseModel):
    label: str
    score: float
    text: str
    source_url: str | None = None


class CommunityExplanation(BaseModel):
    summary: str
    matches: list[EvidenceMatch] = Field(default_factory=list)


class CompetitorMention(BaseModel):
    name: str
    description: str
    evidence: str
    source_url: str
    source_platform: str
    relevance: float


class DiscussionEvidence(BaseModel):
    title: str = Field(max_length=500)
    body: str = Field(default="", max_length=1200)
    source_url: str | None = None


class DiscussionMessageRequest(BaseModel):
    description: str
    platform: str
    community_name: str
    community_url: str | None = None
    community_rules: list[str] = Field(default_factory=list)
    evidence_titles: list[str] = Field(default_factory=list)
    evidence_posts: list[DiscussionEvidence] = Field(default_factory=list)


class DiscussionMessageResponse(BaseModel):
    message: str
    provider: str
    model: str
    human_review_required: bool = True
    rules_verified: bool = False
    review_notice: str


class CompetitorsResponse(BaseModel):
    competitors: list[CompetitorMention]
    status: str


class AnalyzeRequest(BaseModel):
    description: str
    capture_trace: bool = False


class CommunitySignals(BaseModel):
    problem_relevant_posts: int
    evidence_mean: float
    prevalence: float
    avg_recency: float
    avg_comments: float | None = None
    avg_score: float | None = None


class CommunityResult(BaseModel):
    subreddit: str | None = None
    name: str | None = None
    platform: str = "reddit"
    community_kind: str = "community"
    url: str | None = None
    score: float
    posts: list[RedditPost]
    analyzed_posts: int = 0

    # "relevant"    -- qualified via the strong problem_relevant gate.
    # "potential"   -- lower-confidence candidate that cleared the weak gate.
    # "thread_lead" -- topic-relevant indexed thread whose content/date could
    #                  not be independently verified.
    # "suggestion"  -- community directory suggestion without thread evidence.
    status: str = "relevant"

    # Ranking tier used internally before the API filters to verified posts.
    # Potential-tier scores include a ranking penalty; the relevance score in
    # signals remains the unadjusted mean of matching discussion scores.
    rank_tier: str = "relevant"

    # Exposed for explainability: why the community was surfaced.
    signals: CommunitySignals | None = None
    explanation: CommunityExplanation | None = None


class AnalyzeResponse(BaseModel):
    description: str
    communities: list[CommunityResult]
    competitors: list[CompetitorMention] = Field(default_factory=list)
    competitor_discovery_status: str = "not_run"
    sources: dict[str, str] = Field(default_factory=dict)

    # True when none of the returned communities reached the
    # "relevant" tier and the response is showing best-available
    # "potential" candidates instead. Lets the frontend show a
    # "these are lower-confidence matches" banner instead of silently
    # presenting potential-tier results as equally strong.
    fallback_used: bool = False

    # Present only when this request opted into server-side evaluation tracing.
    evaluation_trace_id: str | None = None
