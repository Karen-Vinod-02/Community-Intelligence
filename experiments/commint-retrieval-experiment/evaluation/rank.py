from __future__ import annotations

import math
from collections import defaultdict


def thread_score(
    thread: dict,
) -> float:

    post_score = max(
        int(
            thread.get(
                "score",
                0,
            )
            or 0
        ),
        0,
    )

    comment_count = max(
        int(
            thread.get(
                "num_comments",
                0,
            )
            or 0
        ),
        0,
    )

    useful_comments = [
        c
        for c in thread.get(
            "comments",
            [],
        )
        if len(
            c.get(
                "body",
                "",
            ).strip()
        ) >= 20
    ]

    comment_evidence = sum(
        max(
            int(
                c.get(
                    "score",
                    0,
                )
                or 0
            ),
            0,
        )
        for c in useful_comments
    )

    return (
        math.log1p(post_score)
        + math.log1p(comment_count)
        + 0.25
        * math.log1p(
            comment_evidence
        )
    )


def rank_threads(
    threads: list[dict],
) -> list[dict]:

    ranked = []

    for thread in threads:

        ranked.append(
            {
                **thread,
                "_ranking_score": thread_score(
                    thread
                ),
                "_useful_comment_count": len(
                    [
                        c
                        for c in thread.get(
                            "comments",
                            [],
                        )
                        if len(
                            c.get(
                                "body",
                                "",
                            ).strip()
                        ) >= 20
                    ]
                ),
            }
        )

    ranked.sort(
        key=lambda x: x[
            "_ranking_score"
        ],
        reverse=True,
    )

    return ranked


def rank_communities(
    threads: list[dict],
) -> list[dict]:

    groups = defaultdict(list)

    for thread in threads:

        subreddit = (
            thread.get(
                "subreddit",
                "",
            )
            .lower()
            .strip()
            .replace(
                "r/",
                "",
            )
        )

        if subreddit:
            groups[
                subreddit
            ].append(thread)

    communities = []

    for subreddit, rows in groups.items():

        ranked_threads = rank_threads(
            rows
        )

        communities.append(
            {
                "subreddit": subreddit,
                "thread_count": len(
                    ranked_threads
                ),
                "total_post_score": sum(
                    max(
                        int(
                            t.get(
                                "score",
                                0,
                            )
                            or 0
                        ),
                        0,
                    )
                    for t in ranked_threads
                ),
                "total_comments": sum(
                    int(
                        t.get(
                            "num_comments",
                            0,
                        )
                        or 0
                    )
                    for t in ranked_threads
                ),
                "evidence_comment_count": sum(
                    t.get(
                        "_useful_comment_count",
                        0,
                    )
                    for t in ranked_threads
                ),
                "threads": ranked_threads,
            }
        )

    communities.sort(
        key=lambda c: (
            c[
                "evidence_comment_count"
            ],
            c[
                "thread_count"
            ],
            c[
                "total_comments"
            ],
        ),
        reverse=True,
    )

    return communities