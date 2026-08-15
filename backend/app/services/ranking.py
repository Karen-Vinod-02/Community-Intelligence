from dataclasses import dataclass, field
from app.services.embeddings import EmbeddingService

@dataclass
class CommunityCandidate:
    subreddit: str
    posts: list  #list of RedditPost objects 

class CommunityRanker:

    def __init__(self, top_k_posts: int = 5):
        self.embeddings = EmbeddingService()
        self.top_k_posts = top_k_posts

    def rank(self, description: str, candidates: list[CommunityCandidate], limit: int = 5):
        if not candidates:
            return []

        description_embedding = self.embeddings.encode([description])[0]

        flat_texts, flat_posts, owner_index = [], [], []
        for i, candidate in enumerate(candidates):
            for post in candidate.posts:
                flat_texts.append(f"{post.title}\n{post.body}".strip())
                flat_posts.append(post)
                owner_index.append(i)

        if not flat_texts:
            return []

        embeddings = self.embeddings.encode(flat_texts)

        scored_by_candidate = [[] for _ in candidates]
        for embedding, post, i in zip(embeddings, flat_posts, owner_index):
            score = float(embedding @ description_embedding)
            scored_by_candidate[i].append((score, post))

        results = []
        for candidate, scored_posts in zip(candidates, scored_by_candidate):
            if not scored_posts:
                continue

            scored_posts.sort(key=lambda x: x[0], reverse=True)
            top = scored_posts[: self.top_k_posts]
            community_score = sum(s for s, _ in top) / len(top)

            results.append({
                "subreddit": candidate.subreddit,
                "score": community_score,
                "top_posts": [post for _, post in scored_posts[:3]],  # for the API response
            })

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]