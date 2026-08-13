from app.models.reddit import RedditPost
from app.services.reddit.client import get_reddit_source

class RedditService:

    def __init__(self):
        self.source = get_reddit_source()

    def search_posts(
        self,
        subreddit: str,
        query: str | None = None,
        limit: int = 10,
    ) -> list[RedditPost]:
        return self.source.search_posts(
            subreddit=subreddit,
            query=query,
            limit=limit,
        )