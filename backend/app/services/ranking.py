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
    "action_alignment": 0.20,
    "target_alignment": 0.20,
    "activity_alignment": 0.15,
    "interaction_alignment": 0.15,
    "friction_alignment": 0.20,
    "goal_alignment": 0.10,
}

# NOTE (bugfix): these used to omit action_alignment/target_alignment
# entirely, meaning the directional actor->action->target signal was
# computed but never actually used to decide problem_relevant vs
# weak_or_ambiguous -- only the topical frame_score saw it. That let
# posts pass the evidence gate on activity/interaction overlap alone
# (e.g. shared "developers" + "GitHub" vocabulary) even when the
# action/target relationship didn't match at all. Restored here so
# the gate actually reflects the representation the project is built
# around.
TASK_WEIGHTS = {
    "actor_alignment": 0.10,
    "action_alignment": 0.25,
    "target_alignment": 0.25,
    "activity_alignment": 0.20,
    "interaction_alignment": 0.20,
}

PROBLEM_WEIGHTS = {
    "friction_alignment": 0.65,
    "goal_alignment": 0.35,
}

# NOTE (bugfix): previously a community could be labeled "Relevant"
# off a single thin, borderline post (e.g. community_score ~0.12 from
# 1-of-15 qualifying posts). This floor stops that: a community only
# gets the "relevant" label if its AGGREGATE evidence clears a real
# bar, not just "at least one post technically passed the post-level
# gate." This is a floor on the aggregate score, not a change to any
# post-level threshold.
RELEVANT_COMMUNITY_SCORE_FLOOR = 0.20

# Applied when a community only qualifies through the fallback
# ("potential") tier, so it can never be confused with -- or, when
# scores are close, outrank -- a genuinely "relevant" community.
POTENTIAL_TIER_PENALTY = 0.6


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
    Return the semantic problem representation used for post-level
    comparison.

    Empty fields are intentionally excluded.

    The representation explicitly separates actor, action, and target
    so that directional relationships such as

        small businesses -> find -> designers

    are not collapsed into a single generic topic description.
    """

    frames = {
        "actor_alignment": hypothesis.actor or "",

        "action_alignment": (
            f"{hypothesis.actor or 'people'} "
            f"{hypothesis.action} "
            f"{hypothesis.target}"
            if hypothesis.action and hypothesis.target
            else ""
        ),

        "target_alignment": hypothesis.target or "",

        "activity_alignment": hypothesis.activity or "",

        "interaction_alignment": hypothesis.interaction or "",

        "friction_alignment": hypothesis.friction or "",

        "goal_alignment": hypothesis.goal or "",
    }

    return {
        key: value
        for key, value in frames.items()
        if value.strip()
    }


def _frame_scores(
    hypothesis: ProblemHypothesis,
    post_text: str,
    embeddings,
) -> dict[str, float]:
    """
    Compare a Reddit post against every available dimension
    of the problem representation.

    Only frames that actually exist in the hypothesis are returned.
    """

    frames = _frame_texts(hypothesis)

    if not frames:
        return {}

    texts = [
        post_text,
        *frames.values(),
    ]

    vectors = embeddings.encode(texts)

    post_vector = vectors[0]

    scores = {}

    for index, key in enumerate(frames):
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

    The score is normalized over the frames that actually exist
    in the problem hypothesis.
    """

    weighted_sum = 0.0
    total_weight = 0.0

    for key, score in frame_scores.items():
        weight = FRAME_WEIGHTS.get(key)

        if weight is None:
            continue

        weighted_sum += (
            weight * score
        )

        total_weight += weight

    if total_weight == 0:
        return 0.0

    return weighted_sum / total_weight


def _normalised_weighted_score(
    frame_scores: dict[str, float],
    weights: dict[str, float],
) -> float:
    """
    Compute a weighted average using only dimensions that are
    actually available.
    """

    weighted_sum = 0.0
    total_weight = 0.0

    for key, weight in weights.items():
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

    Task evidence (actor/action/target/activity/interaction)
    establishes what the user is discussing, INCLUDING the
    directional actor->action->target relationship.

    Problem evidence (friction/goal) establishes why the problem
    matters.

    Missing dimensions are excluded from the calculation rather
    than treated as zero-valued evidence.
    """

    task_evidence = _normalised_weighted_score(
        frame_scores,
        TASK_WEIGHTS,
    )

    problem_evidence = _normalised_weighted_score(
        frame_scores,
        PROBLEM_WEIGHTS,
    )

    available_components = []

    if any(
        key in frame_scores
        for key in TASK_WEIGHTS
    ):
        available_components.append(
            task_evidence
        )

    if any(
        key in frame_scores
        for key in PROBLEM_WEIGHTS
    ):
        available_components.append(
            problem_evidence
        )

    if not available_components:
        return 0.0

    if len(available_components) == 2:
        evidence = (
            max(task_evidence, 0.0)
            * max(problem_evidence, 0.0)
        ) ** 0.5
    else:
        evidence = available_components[0]

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

    Thresholds are unchanged from before. What changed is that
    TASK_WEIGHTS now includes action_alignment/target_alignment,
    so the directional actor->action->target signal actually
    participates in this decision instead of only affecting the
    topical frame_score.
    """

    task_evidence = _normalised_weighted_score(
        frame_scores,
        TASK_WEIGHTS,
    )

    problem_evidence = _normalised_weighted_score(
        frame_scores,
        PROBLEM_WEIGHTS,
    )

    evidence_score = _problem_evidence_score(
        frame_scores
    )

    if (
        task_evidence >= 0.50
        and problem_evidence >= 0.40
        and evidence_score >= 0.48
    ):
        status = "problem_relevant"

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


def _aggregate_community(
    qualifying_posts: list[tuple],
    total_posts: int,
) -> tuple[float, float, float]:
    """
    Return (evidence_mean, prevalence, avg_recency) for a set of
    qualifying (score, post, signals) tuples.
    """

    qualifying_count = len(qualifying_posts)

    evidence_mean = (
        sum(item[0] for item in qualifying_posts)
        / qualifying_count
    )

    prevalence = qualifying_count / max(1, total_posts)

    avg_recency = (
        sum(
            item[2]["recency_score"]
            for item in qualifying_posts
        )
        / qualifying_count
    )

    return evidence_mean, prevalence, avg_recency


def _community_score(
    qualifying_posts: list[tuple],
    total_posts: int,
) -> tuple[float, dict]:
    qualifying_count = len(qualifying_posts)

    evidence_mean, prevalence, avg_recency = _aggregate_community(
        qualifying_posts,
        total_posts,
    )

    support_factor = (
        0.5
        + 0.5 * min(1.0, qualifying_count / 2.0)
    )

    evidence_score = (
        evidence_mean
        * support_factor
        * (0.5 + 0.5 * prevalence)
    )

    community_score = (
        0.75 * evidence_score
        + 0.20 * prevalence
        + 0.05 * avg_recency
    )

    signals = {
        "problem_relevant_posts": qualifying_count,
        "evidence_mean": round(evidence_mean, 4),
        "prevalence": round(prevalence, 4),
        "avg_recency": round(avg_recency, 4),
    }

    return community_score, signals


class CommunityRanker:

    def __init__(self, embeddings=None):
        """
        If an embedding service is supplied, reuse it.

        Otherwise create one so the existing CommunityService
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
        trace: dict | None = None,
    ) -> list[dict]:
        """
        Rank candidate communities using problem-relevant posts.

        Qualification is tiered:

            "relevant"  -- has problem_relevant posts AND the
                            aggregated community_score clears
                            RELEVANT_COMMUNITY_SCORE_FLOOR. This is
                            the bar that failed before: a single
                            thin, borderline post used to be enough
                            to earn "Relevant" regardless of how weak
                            the aggregate evidence actually was.
            "potential" -- fallback tier, used ONLY when nothing in
                            this result set reaches "relevant".
                            Pools problem_relevant + weak_or_ambiguous
                            posts together, scored down so it can't
                            be mistaken for -- or outrank -- a
                            genuine "relevant" match.

        This does not force a result count and does not lower any
        post-level threshold. If nothing qualifies under either tier,
        nothing is returned.
        """

        hypotheses = extract_problem_hypotheses(
            description
        )

        if trace is not None:
            trace.clear()
            trace.update({
                "hypothesis_count": len(hypotheses),
                "config": {
                    "min_post_score": MIN_POST_SCORE,
                    "recency_half_life_days": RECENCY_HALF_LIFE_DAYS,
                    "post_relevance_weight": POST_RELEVANCE_WEIGHT,
                    "post_evidence_weight": POST_EVIDENCE_WEIGHT,
                    "post_recency_weight": POST_RECENCY_WEIGHT,
                    "frame_weights": FRAME_WEIGHTS,
                    "task_weights": TASK_WEIGHTS,
                    "problem_weights": PROBLEM_WEIGHTS,
                    "relevant_community_score_floor": RELEVANT_COMMUNITY_SCORE_FLOOR,
                    "potential_tier_penalty": POTENTIAL_TIER_PENALTY,
                    "result_limit": limit,
                },
                "candidate_diagnostics": [],
            })

        if not hypotheses:
            if trace is not None:
                trace.update({"selected_tier": None, "ranked_output": []})
            return []

        per_candidate = []

        for candidate in candidates:

            scored_posts = []
            post_diagnostics = []

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
                    post_diagnostics.append({
                        "post_id": post.id,
                        "score": round(best_score, 6),
                        "signals": best_signals,
                    })

            if not scored_posts:
                if trace is not None:
                    trace["candidate_diagnostics"].append({
                        "subreddit": candidate.subreddit,
                        "input_post_count": len(candidate.posts),
                        "scored_post_count": 0,
                        "posts": [],
                    })
                continue

            total_posts = len(scored_posts)

            strong_posts = [
                item
                for item in scored_posts
                if (
                    item[0] >= MIN_POST_SCORE
                    and item[2]["evidence_status"]
                    == "problem_relevant"
                )
            ]

            weak_posts = [
                item
                for item in scored_posts
                if (
                    item[0] >= MIN_POST_SCORE
                    and item[2]["evidence_status"]
                    == "weak_or_ambiguous"
                )
            ]

            # Diagnostic output (unchanged from before).
            print(
                f"\n[r/{candidate.subreddit}] "
                f"total={total_posts}, "
                f"strong={len(strong_posts)}, "
                f"weak={len(weak_posts)}"
            )

            for score, post, signals in scored_posts:
                print(
                    f"  {score:.3f} | "
                    f"{signals['evidence_status']} | "
                    f"{post.title}"
                )

            per_candidate.append(
                {
                    "subreddit": candidate.subreddit,
                    "total_posts": total_posts,
                    "strong_posts": strong_posts,
                    "weak_posts": weak_posts,
                }
            )

            if trace is not None:
                trace["candidate_diagnostics"].append({
                    "subreddit": candidate.subreddit,
                    "input_post_count": len(candidate.posts),
                    "scored_post_count": total_posts,
                    "strong_post_count": len(strong_posts),
                    "weak_or_ambiguous_post_count": len(weak_posts),
                    "posts": post_diagnostics,
                })

        # ---------------------------------------------------------
        # Pass 1: relevant tier, with the community-score floor.
        # ---------------------------------------------------------

        relevant_results = []

        for entry in per_candidate:
            if not entry["strong_posts"]:
                continue

            entry["strong_posts"].sort(
                key=lambda item: item[0],
                reverse=True,
            )

            community_score, signals = _community_score(
                entry["strong_posts"],
                entry["total_posts"],
            )

            if community_score < RELEVANT_COMMUNITY_SCORE_FLOOR:
                # Technically has a problem_relevant post, but the
                # aggregate is too thin to trust. Falls through to
                # the potential-tier pass instead.
                continue

            top_posts = []
            for score, post, post_signals in entry["strong_posts"][:3]:
                post.evidence_status = post_signals["evidence_status"]
                top_posts.append(post)

            relevant_results.append(
                {
                    "subreddit": entry["subreddit"],
                    "score": round(community_score, 4),
                    "status": "relevant",
                    "top_posts": top_posts,
                    "signals": signals,
                }
            )

        if relevant_results:
            relevant_results.sort(
                key=lambda result: result["score"],
                reverse=True,
            )
            ranked_output = relevant_results[:limit]
            if trace is not None:
                trace.update({
                    "selected_tier": "relevant",
                    "ranked_output": [
                        {
                            "rank": index,
                            "subreddit": item["subreddit"],
                            "score": item["score"],
                            "status": item["status"],
                            "signals": item["signals"],
                            "top_post_ids": [post.id for post in item["top_posts"]],
                        }
                        for index, item in enumerate(ranked_output, start=1)
                    ],
                })
            return ranked_output

        # ---------------------------------------------------------
        # Pass 2: nothing reached "relevant". Fall back to
        # "potential" using strong + weak evidence pooled together,
        # rather than returning zero communities.
        # ---------------------------------------------------------

        potential_results = []

        for entry in per_candidate:
            pooled = entry["strong_posts"] + entry["weak_posts"]

            if not pooled:
                continue

            pooled.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            community_score, signals = _community_score(
                pooled,
                entry["total_posts"],
            )

            community_score *= POTENTIAL_TIER_PENALTY

            top_posts = []
            for score, post, post_signals in pooled[:3]:
                post.evidence_status = post_signals["evidence_status"]
                top_posts.append(post)

            potential_results.append(
                {
                    "subreddit": entry["subreddit"],
                    "score": round(community_score, 4),
                    "status": "potential",
                    "top_posts": top_posts,
                    "signals": signals,
                }
            )

        potential_results.sort(
            key=lambda result: result["score"],
            reverse=True,
        )

        ranked_output = potential_results[:limit]
        if trace is not None:
            trace.update({
                "selected_tier": "potential" if ranked_output else None,
                "ranked_output": [
                    {
                        "rank": index,
                        "subreddit": item["subreddit"],
                        "score": item["score"],
                        "status": item["status"],
                        "signals": item["signals"],
                        "top_post_ids": [post.id for post in item["top_posts"]],
                    }
                    for index, item in enumerate(ranked_output, start=1)
                ],
            })
        return ranked_output
