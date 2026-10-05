from collections import defaultdict

from models import (
    CommunityCandidate,
    ThreadResult,
)


def normalize_subreddit(
    name: str,
) -> str:

    return (
        name
        .strip()
        .removeprefix("r/")
        .lower()
        .strip("/")
    )


def group_communities_by_source(
    candidates: list[
        CommunityCandidate
    ],
):
    """
    Group raw discovery candidates by source.

    IMPORTANT:

    source_rank remains the rank assigned by the
    individual discovery query/source.

    It is NOT treated as a global rank.
    """

    grouped = defaultdict(list)

    for candidate in candidates:

        candidate.subreddit = (
            normalize_subreddit(
                candidate.subreddit
            )
        )

        grouped[
            candidate.source
        ].append(
            candidate
        )

    for source in grouped:

        grouped[source].sort(
            key=lambda x: (
                x.source_rank,
            )
        )

    return grouped


def deduplicate_communities(
    candidates,
):
    """
    Deduplicate communities while preserving all
    source/query evidence.

    Multiple discovery results for the same subreddit
    are retained inside source_records.
    """

    grouped = defaultdict(list)

    for candidate in candidates:

        key = normalize_subreddit(
            candidate.subreddit
        )

        grouped[key].append(
            candidate
        )

    output = []

    for subreddit, items in grouped.items():

        sources = sorted(
            {
                item.source
                for item in items
            }
        )

        source_records = []

        for item in items:

            query = ""

            metadata = getattr(
                item,
                "metadata",
                {},
            )

            if isinstance(
                metadata,
                dict,
            ):
                query = str(
                    metadata.get(
                        "query",
                        "",
                    )
                    or ""
                )

            source_records.append(
                {
                    "source": item.source,
                    "query": query,
                    "source_rank":
                        item.source_rank,
                    "source_score":
                        float(
                            item.source_score
                            or 0
                        ),
                }
            )

        output.append(
            {
                "subreddit": subreddit,
                "name": f"r/{subreddit}",
                "url": (
                    "https://www.reddit.com"
                    f"/r/{subreddit}/"
                ),
                "sources": sources,
                "source_records":
                    source_records,
                "description": next(
                    (
                        item.description
                        for item in items
                        if item.description
                    ),
                    "",
                ),
            }
        )

    return output


def rank_communities(
    communities,
):
    """
    Transparent baseline ranking for the combined
    community candidate pool.

    Ranking signals:

    1. Best discovery confidence
    2. Number of distinct queries that discovered
       the community
    3. Best source rank

    This is intentionally NOT the final CommInt
    relevance ranker.

    It simply gives us a reproducible baseline for
    deciding which communities should proceed to
    expensive thread retrieval.
    """

    ranked = []

    for community in communities:

        records = community.get(
            "source_records",
            [],
        )

        if not records:

            community = {
                **community,
                "best_source_score": 0.0,
                "query_coverage": 0,
                "best_source_rank": float(
                    "inf"
                ),
            }

            ranked.append(
                community
            )

            continue

        scores = []

        ranks = []

        queries = set()

        for record in records:

            try:

                score = float(
                    record.get(
                        "source_score",
                        0,
                    )
                    or 0
                )

            except (
                TypeError,
                ValueError,
            ):

                score = 0.0

            scores.append(
                score
            )

            try:

                rank = int(
                    record.get(
                        "source_rank"
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                rank = 10**9

            ranks.append(
                rank
            )

            query = str(
                record.get(
                    "query",
                    "",
                )
                or ""
            ).strip()

            if query:
                queries.add(
                    query.lower()
                )

        best_source_score = (
            max(scores)
            if scores
            else 0.0
        )

        query_coverage = len(
            queries
        )

        best_source_rank = (
            min(ranks)
            if ranks
            else 10**9
        )

        ranked_community = {
            **community,
            "best_source_score":
                best_source_score,
            "query_coverage":
                query_coverage,
            "best_source_rank":
                best_source_rank,
        }

        ranked.append(
            ranked_community
        )

    ranked.sort(
        key=lambda x: (
            -x[
                "best_source_score"
            ],
            -x[
                "query_coverage"
            ],
            x[
                "best_source_rank"
            ],
            x[
                "subreddit"
            ],
        )
    )

    for index, community in enumerate(
        ranked,
        start=1,
    ):

        community[
            "combined_rank"
        ] = index

    return ranked


def select_top_communities(
    communities,
    limit: int,
):
    """
    Return the first N communities from the
    combined baseline ranking.
    """

    if limit <= 0:
        return []

    return communities[
        :limit
    ]


def deduplicate_threads(
    threads: list[ThreadResult],
):
    """
    Deduplicate threads using:

        subreddit + post_id

    If a thread was discovered through multiple
    community retrieval paths, preserve the union
    of discovery sources.
    """

    grouped = {}

    for thread in threads:

        key = (
            normalize_subreddit(
                thread.subreddit
            ),
            thread.post_id,
        )

        if key not in grouped:

            grouped[key] = thread

            if not thread.discovery_sources:
                thread.discovery_sources = []

        else:

            existing = grouped[key]

            existing.discovery_sources = (
                sorted(
                    set(
                        existing.discovery_sources
                        + thread.discovery_sources
                    )
                )
            )

    output = list(
        grouped.values()
    )

    # Baseline thread ordering.
    #
    # Relevance ranking will be handled separately
    # in the evaluation/ranking stage.
    output.sort(
        key=lambda x: (
            x.num_comments,
            x.score,
        ),
        reverse=True,
    )

    return output