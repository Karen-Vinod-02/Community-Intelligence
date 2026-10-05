from app.services.ranking import CommunityCandidate


class CommunityFilter:
    """
    Reranks candidate communities using semantic evidence from
    their retrieved posts.

    Domain-agnostic.
    """

    def __init__(self, embeddings):
        self.embeddings = embeddings

    @staticmethod
    def _post_text(post) -> str:
        title = getattr(post, "title", "") or ""
        body = getattr(post, "body", "") or ""

        return f"{title} {body}".strip()

    def _semantic_similarity(
        self,
        text_a: str,
        text_b: str,
    ) -> float:
        if not text_a or not text_b:
            return 0.0

        vectors = self.embeddings.encode(
            [
                text_a,
                text_b,
            ]
        )

        similarity = float(
            vectors[0] @ vectors[1]
        )

        return max(
            0.0,
            min(1.0, similarity),
        )

    def _semantic_relevance(
        self,
        description: str,
        posts: list,
    ) -> float:
        texts = [
            self._post_text(post)
            for post in posts
        ]

        texts = [
            text
            for text in texts
            if text
        ]

        if not texts:
            return 0.0

        description_vector = (
            self.embeddings.encode(
                [description]
            )[0]
        )

        post_vectors = (
            self.embeddings.encode(texts)
        )

        similarities = (
            post_vectors @ description_vector
        )

        if len(similarities) == 0:
            return 0.0

        return max(
            0.0,
            min(
                1.0,
                float(max(similarities)),
            ),
        )

    def _problem_frame_score(
        self,
        description: str,
        candidate: CommunityCandidate,
    ) -> float:
        from app.services.query_extraction import (
            extract_problem_hypotheses,
        )

        hypotheses = extract_problem_hypotheses(
            description
        )

        if not hypotheses:
            return 0.0

        posts = candidate.posts

        if not posts:
            return 0.0

        frame_weights = {
            "actor": 0.10,
            "action": 0.20,
            "target": 0.20,
            "activity": 0.15,
            "interaction": 0.15,
            "friction": 0.10,
            "goal": 0.10,
        }

        post_scores = []

        for post in posts:
            post_text = self._post_text(post)

            if not post_text:
                continue

            best_hypothesis_score = 0.0

            for hypothesis in hypotheses:
                frames = {
                    "actor": (
                        hypothesis.actor or ""
                    ),
                    "action": (
                        hypothesis.action or ""
                    ),
                    "target": (
                        hypothesis.target or ""
                    ),
                    "activity": (
                        hypothesis.activity or ""
                    ),
                    "interaction": (
                        hypothesis.interaction or ""
                    ),
                    "friction": (
                        hypothesis.friction or ""
                    ),
                    "goal": (
                        hypothesis.goal or ""
                    ),
                }

                weighted_sum = 0.0
                total_weight = 0.0

                for key, frame in frames.items():
                    if not frame:
                        continue

                    similarity = self._semantic_similarity(
                        frame,
                        post_text,
                    )

                    weight = frame_weights[key]

                    weighted_sum += (
                        weight * similarity
                    )

                    total_weight += weight

                if total_weight == 0:
                    continue

                hypothesis_score = (
                    weighted_sum
                    / total_weight
                )

                best_hypothesis_score = max(
                    best_hypothesis_score,
                    hypothesis_score,
                )

            post_scores.append(
                best_hypothesis_score
            )

        if not post_scores:
            return 0.0

        post_scores.sort(
            reverse=True
        )

        top_scores = post_scores[
            : min(3, len(post_scores))
        ]

        return (
            sum(top_scores)
            / len(top_scores)
        )

    def _community_intent_score(
        self,
        description: str,
        candidate: CommunityCandidate,
    ) -> float:
        posts = candidate.posts

        if not posts:
            return 0.0

        semantic_relevance = (
            self._semantic_relevance(
                description,
                posts,
            )
        )

        problem_frame_relevance = (
            self._problem_frame_score(
                description,
                candidate,
            )
        )

        evidence_density = min(
            1.0,
            len(posts) / 5.0,
        )

        score = (
            0.35 * semantic_relevance
            + 0.55 * problem_frame_relevance
            + 0.10 * evidence_density
        )

        return max(
            0.0,
            min(1.0, score),
        )

    def rerank(
        self,
        description: str,
        candidates: list[CommunityCandidate],
        queries=None,
        trace: dict | None = None,
    ) -> list[CommunityCandidate]:
        """
        Rerank candidate communities using semantic problem
        relevance.

        `queries` is accepted for compatibility with the existing
        CommunityService interface. The semantic reranking itself
        is based on the product description and retrieved evidence.
        """

        scored = []

        for candidate in candidates:
            score = self._community_intent_score(
                description,
                candidate,
            )

            scored.append(
                (
                    score,
                    candidate,
                )
            )

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        if trace is not None:
            trace["config"] = {
                "semantic_relevance_weight": 0.35,
                "problem_frame_relevance_weight": 0.55,
                "evidence_density_weight": 0.10,
                "evidence_density_posts_saturation": 5,
            }
            trace["input_order"] = [candidate.subreddit for candidate in candidates]
            trace["scores"] = [
                {
                    "subreddit": candidate.subreddit,
                    "score": score,
                    "post_ids": [
                        getattr(post, "id", None)
                        for post in candidate.posts
                    ],
                }
                for score, candidate in scored
            ]
            trace["output_order"] = [candidate.subreddit for _, candidate in scored]

        return [
            candidate
            for _, candidate in scored
        ]
