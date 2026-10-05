from __future__ import annotations

import re
from dataclasses import dataclass
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


PRODUCT_WRAPPER_WORDS = {
    "app",
    "application",
    "automation",
    "bot",
    "computer",
    "feature",
    "marketplace",
    "platform",
    "product",
    "service",
    "software",
    "solution",
    "system",
    "technology",
    "tool",
    "website",
}

DESCRIPTION_FRAMING_WORDS = {
    "allows",
    "allowing",
    "based",
    "built",
    "designed",
    "enables",
    "enabling",
    "helps",
    "helping",
    "made",
    "offers",
    "provides",
    "users",
    "using",
    "connecting",
    "connects",
    "lets",
}

METHOD_MARKERS = {
    "from",
    "through",
    "using",
    "with",
    "via",
}

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]*")
_SPACE_RE = re.compile(r"\s+")

_MEANINGFUL_STOP_WORDS = ENGLISH_STOP_WORDS - {
    "find",
    "finding",
    "help",
    "helps",
    "helping",
}


@dataclass(frozen=True)
class ProblemHypothesis:
    """
    Input-derived representation of the problem expressed by a
    product description.

    The representation separates:
        actor -> action -> target

    from the broader activity, interaction, friction and goal fields.
    """

    actor: str | None = None
    action: str | None = None
    target: str | None = None
    activity: str | None = None
    interaction: str | None = None
    friction: str | None = None
    goal: str | None = None
    confidence: float = 1.0
    inferred_fields: tuple[str, ...] = ()

    def evidence_phrases(self) -> dict[str, str]:
        phrases = {}

        if self.actor:
            phrases["actor_alignment"] = (
                f"people like {self.actor}"
            )

        if self.action and self.target:
            phrases["action_alignment"] = (
                f"{self.actor or 'people'} "
                f"{self.action} "
                f"{self.target}"
            )

        if self.target:
            phrases["target_alignment"] = self.target

        if self.activity:
            phrases["activity_alignment"] = (
                f"{self.actor or 'people'} "
                f"trying to {self.activity}"
            )

        if self.interaction:
            phrases["interaction_alignment"] = self.interaction

        if self.friction:
            phrases["friction_alignment"] = self.friction

        if self.goal:
            phrases["goal_alignment"] = self.goal

        return phrases


def _normalise(text: str) -> str:
    return _SPACE_RE.sub(
        " ",
        text.lower(),
    ).strip()


def _words(text: str) -> list[str]:
    result = []

    for raw_word in _WORD_RE.findall(text.lower()):
        word = raw_word.strip("'")

        if (
            len(word) > 2
            and word not in _MEANINGFUL_STOP_WORDS
            and word not in PRODUCT_WRAPPER_WORDS
            and word not in DESCRIPTION_FRAMING_WORDS
            and word not in result
        ):
            result.append(word)

    return result


def _core(words: list[str]) -> str:
    return " ".join(words[:8])


def _add_query(
    queries: list[str],
    seen: set[str],
    query: str,
    max_queries: int,
) -> bool:
    normalised = _normalise(query)

    if not normalised or normalised in seen:
        return False

    seen.add(normalised)
    queries.append(normalised)

    return len(queries) >= max_queries


def _clean_actor(actor: str) -> str:
    actor = _normalise(actor)

    actor = re.sub(
        r"^(an?|the)\s+",
        "",
        actor,
    )

    return actor.strip()


def _clean_target(target: str) -> str:
    target = _normalise(target)

    target = re.sub(
        r"^(an?|the)\s+",
        "",
        target,
    )

    return target.strip()


def _build_hypothesis(
    actor: str,
    action: str,
    target: str,
    confidence: float,
) -> ProblemHypothesis:
    actor = _clean_actor(actor)
    action = _normalise(action)
    target = _clean_target(target)

    activity = f"{action} {target}"

    interaction = (
        f"{actor} {action} {target}"
    )

    friction = (
        f"{actor} struggle to "
        f"{action} {target}"
    )

    goal = (
        f"{actor} successfully "
        f"{action} {target}"
    )

    return ProblemHypothesis(
        actor=actor or None,
        action=action or None,
        target=target or None,
        activity=activity or None,
        interaction=interaction or None,
        friction=friction or None,
        goal=goal or None,
        confidence=confidence,
        inferred_fields=(
            "interaction",
            "friction",
            "goal",
        ),
    )


def _problem_hypotheses(
    description: str,
) -> tuple[ProblemHypothesis, ...]:
    """
    Extract actor/action/target from common grammatical product
    description patterns.

    The parser does not maintain a domain-specific action vocabulary.
    It identifies the action from the grammatical position in the
    sentence.
    """

    normalised = _normalise(description)

    # ------------------------------------------------------------
    # Pattern 1:
    #
    # "connecting local farmers directly with customers"
    #
    # Actor and target are explicitly present around the relationship.
    # ------------------------------------------------------------

    connection_match = re.search(
        r"\bconnecting\s+(.+?)\s+"
        r"(?:directly\s+)?(?:with|to)\s+"
        r"(.+?)(?:[.!?]|$)",
        normalised,
    )

    if connection_match:
        actor = connection_match.group(1)
        target = connection_match.group(2)

        direct = (
            " directly"
            if "directly" in connection_match.group(0)
            else ""
        )

        hypothesis = ProblemHypothesis(
            actor=_clean_actor(actor),
            action="connect",
            target=_clean_target(target),
            activity=(
                f"connect with {_clean_target(target)}"
            ),
            interaction=(
                f"{_clean_actor(actor)} "
                f"connect{direct} with "
                f"{_clean_target(target)}"
            ),
            friction=(
                f"{_clean_actor(actor)} struggle to "
                f"connect with {_clean_target(target)}"
            ),
            goal=(
                f"{_clean_actor(actor)} successfully "
                f"connect with {_clean_target(target)}"
            ),
            confidence=0.75,
            inferred_fields=(
                "activity",
                "friction",
                "goal",
            ),
        )

        return (hypothesis,)

    # ------------------------------------------------------------
    # Pattern 2:
    #
    # "helping students find reliable study partners"
    #
    # Crucially, the token immediately following "helping <actor>"
    # is treated as the action. No action vocabulary is required.
    # ------------------------------------------------------------

    helping_match = re.search(
        r"\bhelping\s+(.+?)\s+"
        r"([a-zA-Z][a-zA-Z'-]*)\s+"
        r"(.+?)(?:[.!?]|$)",
        normalised,
    )

    if helping_match:
        actor = helping_match.group(1)
        action = helping_match.group(2)
        target = helping_match.group(3)

        return (
            _build_hypothesis(
                actor=actor,
                action=action,
                target=target,
                confidence=1.0,
            ),
        )

    # ------------------------------------------------------------
    # Pattern 3:
    #
    # "helps students find reliable study partners"
    # "help students find reliable study partners"
    # ------------------------------------------------------------

    helps_match = re.search(
        r"\bhelp(?:s)?\s+(.+?)\s+"
        r"([a-zA-Z][a-zA-Z'-]*)\s+"
        r"(.+?)(?:[.!?]|$)",
        normalised,
    )

    if helps_match:
        actor = helps_match.group(1)
        action = helps_match.group(2)
        target = helps_match.group(3)

        return (
            _build_hypothesis(
                actor=actor,
                action=action,
                target=target,
                confidence=1.0,
            ),
        )

    # ------------------------------------------------------------
    # Pattern 4:
    #
    # "enabling developers to understand large codebases"
    # "allows developers to understand large codebases"
    # "lets developers understand large codebases"
    #
    # The action is again obtained from its grammatical position.
    # ------------------------------------------------------------

    enabling_match = re.search(
        r"\b(?:enabling|enables|allowing|allows|letting|lets)\s+"
        r"(.+?)\s+"
        r"(?:to\s+)?"
        r"([a-zA-Z][a-zA-Z'-]*)\s+"
        r"(.+?)(?:[.!?]|$)",
        normalised,
    )

    if enabling_match:
        actor = enabling_match.group(1)
        action = enabling_match.group(2)
        target = enabling_match.group(3)

        return (
            _build_hypothesis(
                actor=actor,
                action=action,
                target=target,
                confidence=0.95,
            ),
        )

    # ------------------------------------------------------------
    # Pattern 5:
    #
    # "for developers to understand large codebases"
    #
    # This is a weaker fallback because the actor/action relationship
    # is less explicitly tied to a product construction pattern.
    # ------------------------------------------------------------

    for_match = re.search(
        r"\bfor\s+(.+?)\s+"
        r"(?:to\s+)?"
        r"([a-zA-Z][a-zA-Z'-]*)\s+"
        r"(.+?)(?:[.!?]|$)",
        normalised,
    )

    if for_match:
        actor = for_match.group(1)
        action = for_match.group(2)
        target = for_match.group(3)

        return (
            _build_hypothesis(
                actor=actor,
                action=action,
                target=target,
                confidence=0.70,
            ),
        )

    return ()


def extract_problem_hypotheses(
    description: str,
) -> tuple[ProblemHypothesis, ...]:
    if (
        not isinstance(description, str)
        or not description.strip()
    ):
        return ()

    return _problem_hypotheses(description)


def _relationship_phrases(
    hypothesis: ProblemHypothesis,
) -> tuple[str, ...]:
    """
    Generate general linguistic reformulations from the extracted
    actor/action/target relationship.

    No action-specific vocabulary is used.
    """

    actor = hypothesis.actor
    action = hypothesis.action
    target = hypothesis.target

    if not actor or not action or not target:
        return ()

    return (
        f"{actor} {action} {target}",
        f"{actor} trying to {action} {target}",
        f"{actor} looking for ways to {action} {target}",
        f"{actor} struggling to {action} {target}",
        f"how {actor} {action} {target}",
        f"{actor} need help {action} {target}",
    )


def extract_search_queries(
    description: str,
    max_queries: int = 8,
) -> list[str]:
    """
    Produce discovery queries from the extracted problem structure.

    Queries prioritize the explicit actor/action/target relationship,
    then fall back to broader concept-level formulations.
    """

    if (
        not isinstance(description, str)
        or max_queries <= 0
    ):
        return []

    normalised = _normalise(description)
    words = _words(normalised)

    if not words:
        return []

    hypotheses = extract_problem_hypotheses(
        description
    )

    hypothesis = (
        hypotheses[0]
        if hypotheses
        else None
    )

    queries = []
    seen = set()

    if hypothesis:
        for query in _relationship_phrases(
            hypothesis
        ):
            if _add_query(
                queries,
                seen,
                query,
                max_queries,
            ):
                return queries

    concept = _core(words)

    fallback_queries = (
        f"{concept} community",
        f"{concept} discussion",
        f"people looking for {concept}",
        f"people struggling with {concept}",
        f"how people {concept}",
        f"ways to {concept}",
        f"{concept} advice",
        f"{concept} experiences",
    )

    for query in fallback_queries:
        if _add_query(
            queries,
            seen,
            query,
            max_queries,
        ):
            break

    return queries