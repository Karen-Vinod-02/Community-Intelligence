from __future__ import annotations

from dataclasses import asdict
from typing import Any


def serialize_result(
    result: dict[str, Any],
) -> dict[str, Any]:

    output = {
        key: value
        for key, value in result.items()
        if key != "queries"
    }

    output["queries"] = []

    for query_result in result.get(
        "queries",
        [],
    ):

        serialized = {
            key: value
            for key, value in query_result.items()
            if key != "threads"
        }

        serialized["threads"] = [
            asdict(thread)
            for thread in query_result.get(
                "threads",
                [],
            )
        ]

        output["queries"].append(
            serialized
        )

    return output


def flatten_unique_threads(
    result: dict[str, Any],
) -> list[dict[str, Any]]:

    unique = {}

    for query_result in result.get(
        "queries",
        [],
    ):

        for thread in query_result.get(
            "threads",
            [],
        ):

            post_id = thread.get(
                "post_id"
            )

            if not post_id:
                continue

            if post_id not in unique:

                unique[
                    post_id
                ] = thread

            else:

                # Keep the richer version
                existing = unique[
                    post_id
                ]

                if len(
                    thread.get(
                        "comments",
                        [],
                    )
                ) > len(
                    existing.get(
                        "comments",
                        [],
                    )
                ):

                    unique[
                        post_id
                    ] = thread

    return list(
        unique.values()
    )