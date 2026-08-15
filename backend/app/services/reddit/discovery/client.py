import os

from app.services.reddit.discovery.base import DiscoverySource
from app.services.reddit.discovery.websearch import WebSearchDiscoverySource

def get_discovery_source() -> DiscoverySource:

    provider = os.getenv(
        "REDDIT_DISCOVERY_SOURCE",
        "websearch",
    ).lower()

    if provider == "websearch":
        return WebSearchDiscoverySource()

    raise ValueError(
        f"Unsupported Reddit discovery source: {provider}"
    )