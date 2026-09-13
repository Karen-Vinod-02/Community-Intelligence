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
    "based",
    "built",
    "designed",
    "enables",
    "helps",
    "helping",
    "made",
    "offers",
    "provides",
    "users",
    "using",
    "connecting",
}

ACTION_WORDS = {
    "buy",
    "create",
    "debug",
    "develop",
    "find",
    "generate",
    "get",
    "learn",
    "make",
    "manage",
    "organize",
    "plan",
    "produce",
    "sell",
    "share",
    "solve",
    "write",
}

METHOD_MARKERS = {"from", "through", "using", "with", "via"}
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]*")
_SPACE_RE = re.compile(r"\s+")
_MEANINGFUL_STOP_WORDS = ENGLISH_STOP_WORDS - {"find", "finding"}


@dataclass(frozen=True)
class ProblemHypothesis:
    """An input-derived hypothesis about the difficulty a product addresses."""

    actor: str | None = None
    activity: str | None = None
    interaction: str | None = None
    friction: str | None = None
    goal: str | None = None
    confidence: float = 1.0
    inferred_fields: tuple[str, ...] = ()

    def evidence_phrases(self) -> dict[str, str]:
        phrases = {}
        if self.actor:
            phrases["actor_alignment"] = f"people like {self.actor}"
        if self.activity:
            phrases["activity_alignment"] = (
                f"{self.actor or 'people'} trying to {self.activity}"
            )
        if self.interaction:
            phrases["interaction_alignment"] = self.interaction
        if self.friction:
            phrases["friction_alignment"] = self.friction
        if self.goal:
            phrases["goal_alignment"] = self.goal
        return phrases


def _problem_hypotheses(description: str) -> tuple[ProblemHypothesis, ...]:
    """Derive generic problem hypotheses from grammatical product patterns.

    This intentionally uses only relationship patterns present in the input.
    Missing problem details remain absent or receive lower confidence rather
    than being filled from a domain ontology.
    """

    normalised = _normalise(description)
    connection_match = re.search(
        r"\bconnecting\s+(.+?)\s+(?:directly\s+)?(?:with|to)\s+(.+?)(?:[.!?]|$)",
        normalised,
    )
    if connection_match:
        actor = connection_match.group(1).strip()
        counterparty = connection_match.group(2).strip()
        direct = " directly" if " directly " in f" {normalised} " else ""
        return (
            ProblemHypothesis(
                actor=actor,
                interaction=(
                    f"{actor} connect{direct} with {counterparty} "
                    f"or find people to buy from them"
                ),
                activity=(
                    f"sell or exchange products or services with {counterparty}"
                ),
                friction=f"{actor} struggle to find buyers or reach {counterparty}",
                goal=f"{actor} sell directly to {counterparty}",
                confidence=0.75,
                inferred_fields=("activity", "friction", "goal"),
            ),
        )

    helping_match = re.search(
        r"\bhelp(?:s|ing)?\s+(.+?)\s+(?:to\s+)?"
        r"(find|understand|learn|buy|sell|create|manage|organize|"
        r"develop|write|solve|plan|get|make)\s+(.+?)(?:[.!?]|$)",
        normalised,
    )
    if helping_match:
        actor = helping_match.group(1).strip()
        action = helping_match.group(2).strip()
        target = helping_match.group(3).strip()
        return (
            ProblemHypothesis(
                actor=actor,
                activity=f"{action} {target}",
                interaction=(
                    f"{actor} connect with {target}"
                    if action == "find"
                    else None
                ),
                friction=f"{actor} struggle to {action} {target}",
                goal=f"{actor} reliably {action} {target}",
                confidence=1.0,
            ),
        )

    words = _words(normalised)
    actions = [index for index, word in enumerate(words) if word in ACTION_WORDS]
    if actions:
        action_index = actions[0]
        actor = " ".join(words[:action_index]).strip() or None
        target = " ".join(words[action_index + 1 :]).strip() or None
        if actor and target:
            action = words[action_index]
            return (
                ProblemHypothesis(
                    actor=actor,
                    activity=f"{action} {target}",
                    friction=f"{actor} struggle to {action} {target}",
                    goal=f"{actor} successfully {action} {target}",
                    confidence=0.7,
                    inferred_fields=("friction", "goal"),
                ),
            )
    return ()


def extract_problem_hypotheses(
    description: str,
) -> tuple[ProblemHypothesis, ...]:
    """Return confidence-weighted, input-derived problem hypotheses."""

    if not isinstance(description, str) or not description.strip():
        return ()
    return _problem_hypotheses(description)


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


def _relationship_phrases(description: str, words: list[str]) -> tuple[str, ...]:
    """Extract actor/action/object/method phrases when the wording supports it."""

    connection_match = re.search(
        r"\bconnecting\s+(.+?)\s+(?:with|to)\s+(.+?)(?:[.!?]|$)",
        description.lower(),
    )
    if connection_match:
        audience = connection_match.group(2)
        looking_match = re.match(r"(.+?)\s+looking\s+for\s+(.+)", audience)
        if looking_match:
            actor = _normalise(looking_match.group(1))
            task = _normalise(looking_match.group(2))
            return (
                f"{actor} looking for {task}",
                f"{actor} finding {task}",
                f"where can {actor} find {task}",
                f"{actor} struggling to find {task}",
            )

    actions = [index for index, word in enumerate(words) if word in ACTION_WORDS]
    if not actions:
        return ()

    action_index = actions[0]
    actor_words = words[:action_index]
    while actor_words and actor_words[0] in {"ai", "online", "digital"}:
        actor_words.pop(0)
    actor = " ".join(actor_words)
    remainder = words[action_index + 1 :]
    if not actor or not remainder:
        return ()

    method_index = next(
        (index for index, word in enumerate(remainder) if word in METHOD_MARKERS),
        len(remainder),
    )
    task = " ".join(remainder[:method_index])
    method = " ".join(remainder[method_index + 1 :])
    action = words[action_index]
    phrases = [
        f"{actor} {action} {task}",
        f"{actor} struggling to {action} {task}",
        f"how {actor} {action} {task}",
        f"{actor} looking for ways to {action} {task}",
    ]
    if method:
        phrases.append(f"{actor} {action} {task} {method}")
    return tuple(phrases)


def _normalise(query: str) -> str:
    return _SPACE_RE.sub(" ", query.lower()).strip()


def extract_search_queries(description: str, max_queries: int = 8) -> list[str]:
    """Return input-dependent, Reddit-style problem queries.

    Relationships are preferred over generic suffix templates.  When a
    description contains no recoverable action relationship, the meaningful
    concept sequence remains the deterministic fallback.
    """

    if not isinstance(description, str) or max_queries <= 0:
        return []

    words = _words(_SPACE_RE.sub(" ", description).strip())
    concept = _core(words)
    if not concept:
        return []

    relationship_queries = _relationship_phrases(description, words)
    # These operators target community pages and discussions, rather than
    # repeating the product description as a keyword bag.
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

    queries = []
    seen = set()
    for query in (*relationship_queries, *fallback_queries):
        normalised = _normalise(query)
        if normalised and normalised not in seen:
            seen.add(normalised)
            queries.append(normalised)
        if len(queries) >= max_queries:
            break
    return queries
