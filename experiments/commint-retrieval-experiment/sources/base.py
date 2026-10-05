from abc import ABC, abstractmethod

from models import (
    CommunityCandidate,
    ThreadResult,
)


class CommunityDiscoverySource(ABC):

    name: str

    @abstractmethod
    def discover(
        self,
        case: dict,
    ) -> list[CommunityCandidate]:
        raise NotImplementedError


class ThreadRetrievalSource(ABC):

    name: str

    @abstractmethod
    def retrieve_threads(
        self,
        case: dict,
        subreddit: str,
    ) -> list[ThreadResult]:
        raise NotImplementedError