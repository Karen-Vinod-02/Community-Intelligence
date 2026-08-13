from abc import ABC, abstractmethod
from app.models.reddit import RedditPost, RedditComment

class RedditSource(ABC):

    @abstractmethod
    def search_posts(
        self,
        subreddit: str,
        query: str | None = None,
        limit: int = 50,
    ) -> list[RedditPost]:
        raise NotImplementedError

    @abstractmethod
    def fetch_comments(
        self,
        post_id: str,
        limit: int = 100,
    ) -> list[RedditComment]:
        raise NotImplementedError