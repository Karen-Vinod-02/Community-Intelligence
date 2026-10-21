from abc import ABC, abstractmethod


class CommunityDiscoveryResults(list):
    def __init__(self, communities: list[str], evidence: dict | None = None):
        super().__init__(communities)
        self.evidence = evidence or {}


class DiscoverySource(ABC):

    @abstractmethod
    def discover_communities(
        self,
        query: str,
        limit: int = 10,
    ) -> list[str]:
        raise NotImplementedError