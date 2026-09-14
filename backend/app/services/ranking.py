import time
from dataclasses import dataclass

from app.models.reddit import RedditPost
from app.services.query_extraction import (
    ProblemHypothesis,
    extract_problem_hypotheses,
)


MIN_POST_SCORE = 0.25
RECENCY_HALF_LIFE_DAYS = 30

POST_RELEVANCE_WEIGHT = 0.75
POST_EVIDENCE_WEIGHT = 0.20
POST_RECENCY_WEIGHT = 0.05

FRAME_WEIGHTS = {
    "actor_alignment": 0.10,
    "activity_alignment": 0.20,
    "interaction_alignment": 0.25,
    "friction_alignment": 0.30,
    "goal_alignment": 0.15,
}


@dataclass
class CommunityCandidate:
    subreddit: str
    posts: list[RedditPost]
    queries: tuple[str, ...] = ()


def _semantic_similarity(
    text_a: str,
    text_b: str,
    embeddings,
) -> float:
    """
    Compute cosine similarity using normalized sentence embeddings.
    """

    if not text_a or not text_b:
        return 0.0

    vectors = embeddings.encode(
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


def _recency_score(post: RedditPost) -> float:
    """
    Give newer posts more weight using exponential decay.
    """

    if post.created_utc is None:
        return 0.0

    age_seconds = max(
        0.0,
        time.time() - post.created_utc,
    )

    age_days = age_seconds / 86400.0

    return 0.5 ** (
        age_days / RECENCY_HALF_LIFE_DAYS
    )


def _frame_texts(
    hypothesis: ProblemHypothesis,
) -> dict[str, str]:
    """
    Return the semantic problem representation used for post-level comparison.
    """

    return {
        "actor_alignment": hypothesis.actor or "",
        "activity_alignment": hypothesis.activity or "",
        "interaction_alignment": hypothesis.interaction or "",
        "friction_alignment": hypothesis.friction or "",
        "goal_alignment": hypothesis.goal or "",
    }


def _frame_scores(
    hypothesis: ProblemHypothesis,
    post_text: str,
    embeddings,
) -> dict[str, float]:
    """
    Compare a Reddit post independently against every dimension
    of the problem representation.
    """

    frames = _frame_texts(hypothesis)

    non_empty_frames = {
        key: value
        for key, value in frames.items()
        if value
    }

    if not non_empty_frames:
        return {
            key: 0.0
            for key in frames
        }

    texts = [
        post_text,
        *non_empty_frames.values(),
    ]

    vectors = embeddings.encode(texts)

    post_vector = vectors[0]

    scores = {
        key: 0.0
        for key in frames
    }

    for index, key in enumerate(
        non_empty_frames
    ):
        similarity = float(
            post_vector @ vectors[index + 1]
        )

        scores[key] = max(
            0.0,
            min(1.0, similarity),
        )

    return scores


def _problem_frame_score(
    frame_scores: dict[str, float],
) -> float:
    """
    Produce an overall problem-frame similarity score.

    The score is weighted toward friction and interaction because
    these dimensions distinguish the target problem from generic
    topical discussion.
    """

    weighted_sum = 0.0
    total_weight = 0.0

    for key, weight in FRAME_WEIGHTS.items():
        if key not in frame_scores:
            continue

        weighted_sum += (
            weight * frame_scores[key]
        )
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return weighted_sum / total_weight


def _problem_evidence_score(
    frame_scores: dict[str, float],
) -> float:
    """
    Measure whether the post provides evidence of the actual
    problem rather than merely sharing the same topic.

    Activity and interaction establish what the user is discussing.
    Friction & goal establish why the problem matters.

    Friction receives the strongest weight because it is the most
    direct representation of the underlying problem.
    """

    activity = frame_scores.get(
        "activity_alignment",
        0.0,
    )

    interaction = frame_scores.get(
        "interaction_alignment",
        0.0,
    )

    friction = frame_scores.get(
        "friction_alignment",
        0.0,
    )

    goal = frame_scores.get(
        "goal_alignment",
        0.0,
    )

    task_evidence = (
        0.45 * activity
        + 0.55 * interaction
    )

    problem_evidence = (
        0.65 * friction
        + 0.35 * goal
    )

    # geometric mean prevents one side from dominating.
    evidence = (
        max(task_evidence, 0.0)
        * max(problem_evidence, 0.0)
    ) ** 0.5

    return max(
        0.0,
        min(1.0, evidence),
    )


def _evidence_status(
    frame_scores: dict[str, float],
) -> tuple[str, dict]:
    """
    Classify the strength of problem evidence using semantic
    relationships between the post and the problem frame.

    Task evidence establishes that the post concerns the relevant
    activity or interaction.

    Problem evidence provides additional support that the post
    reflects the underlying need, difficulty, and/or desired outcome.

    The two dimensions are intentionally not required to both exceed
    a high threshold because short Reddit posts may express the
    problem implicitly.
    """

    activity = frame_scores.get(
        "activity_alignment",
        0.0,
    )

    interaction = frame_scores.get(
        "interaction_alignment",
        0.0,
    )

    friction = frame_scores.get(
        "friction_alignment",
        0.0,
    )

    goal = frame_scores.get(
        "goal_alignment",
        0.0,
    )

    task_evidence = (
        0.45 * activity
        + 0.55 * interaction
    )

    problem_evidence = (
        0.65 * friction
        + 0.35 * goal
    )

    evidence_score = _problem_evidence_score(
        frame_scores
    )

    # Strong task evidence + supporting problem evidence.
    if (
        task_evidence >= 0.50
        and problem_evidence >= 0.40
        and evidence_score >= 0.48
    ):
        status = "problem_relevant"

    # A post can still be useful when it strongly represents the problem but does not explicitly express every dimension.
    elif (
        task_evidence >= 0.58
        and problem_evidence >= 0.30
    ):
        status = "problem_relevant"

    elif (
        task_evidence >= 0.38
        or problem_evidence >= 0.35
    ):
        status = "weak_or_ambiguous"

    else:
        status = "topic_only"

    diagnostics = {
        "task_evidence": round(
            task_evidence,
            4,
        ),
        "problem_evidence": round(
            problem_evidence,
            4,
        ),
        "evidence_score": round(
            evidence_score,
            4,
        ),
    }

    return status, diagnostics


def _post_score(
    hypothesis: ProblemHypothesis,
    post: RedditPost,
    embeddings,
) -> tuple[float, dict]:
    """
    Score one Reddit post against one problem hypothesis.
    """

    post_text = (
        f"{post.title} {post.body}"
    ).strip()

    if not post_text:
        return 0.0, {
            "frame_score": 0.0,
            "evidence_score": 0.0,
            "recency_score": 0.0,
            "evidence_status": "topic_only",
            "frame_scores": {},
        }

    frame_scores = _frame_scores(
        hypothesis,
        post_text,
        embeddings,
    )

    frame_score = _problem_frame_score(
        frame_scores
    )

    evidence_score = _problem_evidence_score(
        frame_scores
    )

    recency = _recency_score(post)

    evidence_status, evidence_diagnostics = (
        _evidence_status(frame_scores)
    )

    # Problem relevance is the primary ranking signal.
    #
    # Evidence contributes independently so that two posts with
    # similar topical similarity can still be separated based on
    # how strongly they represent the actual problem.
    score = (
        POST_RELEVANCE_WEIGHT * frame_score
        + POST_EVIDENCE_WEIGHT * evidence_score
        + POST_RECENCY_WEIGHT * recency
    )

    signals = {
        "frame_score": round(
            frame_score,
            4,
        ),
        "evidence_score": round(
            evidence_score,
            4,
        ),
        "recency_score": round(
            recency,
            4,
        ),
        "evidence_status": evidence_status,
        "frame_scores": {
            key: round(
                value,
                4,
            )
            for key, value in frame_scores.items()
        },
        **evidence_diagnostics,
    }

    return (
        max(
            0.0,
            min(1.0, score),
        ),
        signals,
    )


class CommunityRanker:
    def __init__(self, embeddings=None):
        """
        If an embedding service is supplied, reuse it; else create one so the existing CommunityService
        constructor, which calls CommunityRanker(), remains
        compatible.
        """

        if embeddings is None:
            from app.services.embeddings import EmbeddingService

            embeddings = EmbeddingService()

        self.embeddings = embeddings

    def rank(
        self,
        description: str,
        candidates: list[CommunityCandidate],
        limit: int = 5,
        context: str = "",
    ) -> list[dict]:
        """
        Rank candidate communities using problem-relevant posts.

        A community becomes recommendable only when at least one
        retrieved post provides sufficient semantic evidence of
        the represented problem.
        """

        hypotheses = extract_problem_hypotheses(
            description
        )

        if not hypotheses:
            return []

        results = []

        for candidate in candidates:
            scored_posts = []

            for post in candidate.posts:
                best_score = 0.0
                best_signals = None

                for hypothesis in hypotheses:
                    score, signals = _post_score(
                        hypothesis,
                        post,
                        self.embeddings,
                    )

                    if (
                        best_signals is None
                        or score > best_score
                    ):
                        best_score = score
                        best_signals = signals

                if best_signals is not None:
                    scored_posts.append(
                        (
                            best_score,
                            post,
                            best_signals,
                        )
                    )

            if not scored_posts:
                continue

            qualifying_posts = [
                item
                for item in scored_posts
                if (
                    item[0] >= MIN_POST_SCORE
                    and item[2]["evidence_status"]
                    == "problem_relevant"
                )
            ]

            total_posts = len(
                scored_posts
            )

            qualifying_count = len(
                qualifying_posts
            )
            # Diagnostic 
            print(
                f"\n[r/{candidate.subreddit}] "
                f"total={total_posts}, "
                f"qualifying={qualifying_count}"
            )

            for score, post, signals in scored_posts:
                print(
                    f"  {score:.3f} | "
                    f"{signals['evidence_status']} | "
                    f"{post.title}"
                )
            # No actual problem evidence means the community should not be recommended.
            if qualifying_count == 0:
                continue

            qualifying_posts.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            evidence_mean = (
                sum(
                    item[0]
                    for item in qualifying_posts
                )
                / qualifying_count
            )

            prevalence = (
                qualifying_count
                / max(
                    1,
                    total_posts,
                )
            )

            avg_recency = (
                sum(
                    item[2]["recency_score"]
                    for item in qualifying_posts
                )
                / qualifying_count
            )

            # Communities supported by multiple problem-relevant
            # posts receive more confidence than communities with
            # only 1 accidental match.
            support_factor = (
                0.5
                + 0.5
                * min(
                    1.0,
                    qualifying_count / 2.0,
                )
            )

            evidence_score = (
                evidence_mean
                * support_factor
                * (
                    0.5
                    + 0.5 * prevalence
                )
            )

            community_score = (
                0.75 * evidence_score
                + 0.20 * prevalence
                + 0.05 * avg_recency
            )

            top_posts = []

            for (
                score,
                post,
                signals,
            ) in qualifying_posts[:3]:

                post.evidence_status = (
                    signals["evidence_status"]
                )

                top_posts.append(post)

            results.append(
                {
                    "subreddit": candidate.subreddit,
                    "score": round(
                        community_score,
                        4,
                    ),
                    "top_posts": top_posts,
                    "signals": {
                        "total_posts": total_posts,
                        "problem_relevant_posts": (
                            qualifying_count
                        ),
                        "evidence_mean": round(
                            evidence_mean,
                            4,
                        ),
                        "prevalence": round(
                            prevalence,
                            4,
                        ),
                        "avg_recency": round(
                            avg_recency,
                            4,
                        ),
                        "post_signals": [
                            {
                                "post_id": post.id,
                                "score": round(
                                    score,
                                    4,
                                ),
                                **signals,
                            }
                            for (
                                score,
                                post,
                                signals,
                            )
                            in scored_posts
                        ],
                    },
                }
            )

        results.sort(
            key=lambda result: result["score"],
            reverse=True,
        )

        return results[:limit]