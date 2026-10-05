from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CommunityCandidate:
    case_id: str
    subreddit: str
    source: str

    source_rank: int
    source_score: float = 0.0

    name: str = ""
    description: str = ""
    url: str = ""

    discovery_reason: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class ThreadResult:
    case_id: str
    subreddit: str
    post_id: str

    source: str

    source_rank: int = 0
    source_score: float = 0.0

    title: str = ""
    text: str = ""
    url: str = ""

    author: Optional[str] = None
    created_at: Optional[str] = None

    score: float = 0.0
    num_comments: int = 0

    query: str = ""

    comments: list[dict[str, Any]] = field(
        default_factory=list
    )

    discovery_sources: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )