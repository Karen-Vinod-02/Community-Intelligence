from abc import ABC, abstractmethod

class DiscoverySource(ABC):

    @abstractmethod
    def discover_communities(
        self,
        query: str,
        limit: int = 10,
    ) -> list[str]:
        raise NotImplementedError