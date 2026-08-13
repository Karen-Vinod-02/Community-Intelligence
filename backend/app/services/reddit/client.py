import os

from app.services.reddit.base import RedditSource
from app.services.reddit.arcticshift import ArcticShiftSource


def get_reddit_source() -> RedditSource:
    provider = os.getenv("REDDIT_SOURCE", "arcticshift").lower()

    if provider == "arcticshift":
        return ArcticShiftSource()

    raise ValueError(
        f"Unsupported Reddit source: {provider}"
    )