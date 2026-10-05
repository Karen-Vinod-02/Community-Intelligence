
from pathlib import Path
import json


def save_json(
    path: Path,
    data,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _format_metric(value):
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _format_coverage(metrics):
    annotated = metrics.get("annotated_items", 0)
    total = metrics.get("n_items", 0)

    return f"{annotated}/{total}"


def _format_status(metrics):
    if metrics.get("judgments_complete") is True:
        return "Complete"

    return "Incomplete"


def _write_source_table(
    lines,
    metrics,
):
    """
    Writes a table for one or more sources.

    Expected structure:

    {
        "reddit_mcp": {
            "study_accountability": {
                "n_items": 14,
                "annotated_items": 14,
                "unannotated_items": 0,
                "judgments_complete": True,
                "ndcg@10": 0.99,
                "mrr": 1.0
            }
        }
    }
    """

    if not metrics:
        lines.extend(
            [
                "No metrics available.",
                "",
            ]
        )
        return

    lines.extend(
        [
            "| Source | Evaluation | Judged | nDCG@10 | MRR | Status |",
            "|---|---|---:|---:|---:|---|",
        ]
    )

    for source, evaluations in metrics.items():

        if not isinstance(evaluations, dict):
            continue

        for evaluation, result in evaluations.items():

            if not isinstance(result, dict):
                continue

            lines.append(
                "| "
                f"{source} | "
                f"{evaluation} | "
                f"{_format_coverage(result)} | "
                f"{_format_metric(result.get('ndcg@10'))} | "
                f"{_format_metric(result.get('mrr'))} | "
                f"{_format_status(result)} |"
            )

    lines.append("")


def _write_thread_table(
    lines,
    metrics,
):
    """
    Writes thread retrieval metrics.

    Expected structure:

    {
        "study_accountability": {
            "n_items": 15,
            "annotated_items": 15,
            "unannotated_items": 0,
            "judgments_complete": True,
            "ndcg@10": 0.8356,
            "mrr": 1.0
        }
    }
    """

    if not metrics:
        lines.extend(
            [
                "No metrics available.",
                "",
            ]
        )
        return

    lines.extend(
        [
            "| Evaluation | Judged | nDCG@10 | MRR | Status |",
            "|---|---:|---:|---:|---|",
        ]
    )

    for evaluation, result in metrics.items():

        if not isinstance(result, dict):
            continue

        lines.append(
            "| "
            f"{evaluation} | "
            f"{_format_coverage(result)} | "
            f"{_format_metric(result.get('ndcg@10'))} | "
            f"{_format_metric(result.get('mrr'))} | "
            f"{_format_status(result)} |"
        )

    lines.append("")


def save_report(
    path: Path,
    results: dict,
):
    lines = [
        "# CommInt Retrieval Experiment",
        "",
        (
            "Metrics are calculated only when all retrieved items "
            "in an evaluation are annotated. Incomplete judgments "
            "are reported as N/A rather than treating unjudged "
            "items as irrelevant."
        ),
        "",
    ]

    # ------------------------------------------------
    # CURRENT EVALUATOR STRUCTURE
    #
    # results = {
    #     "communities": {
    #         source: {
    #             evaluation: metrics
    #         }
    #     },
    #     "threads": {
    #         evaluation: metrics
    #     }
    # }
    # ------------------------------------------------

    if "communities" in results:

        lines.extend(
            [
                "## Community Retrieval",
                "",
            ]
        )

        _write_source_table(
            lines,
            results["communities"],
        )

    if "threads" in results:

        lines.extend(
            [
                "## Thread Retrieval",
                "",
            ]
        )

        _write_thread_table(
            lines,
            results["threads"],
        )

    if results.get("community_scope", {}).get("agent_reach_safety_scoped"):
        lines.extend(
            [
                "## Agent-Reach Safety Scope",
                "",
                (
                    "Safety eligibility is reported separately from relevance. "
                    "These post-hit records do not provide a global ranked "
                    "community list, so ranking metrics are not calculated."
                ),
                "",
                "| Evaluation | Raw post rows | Distinct communities | Safety eligible | Safety ineligible | Eligible judged | Judged relevance distribution | Ranking metrics |",
                "|---|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for evaluation, scope in results["community_scope"]["agent_reach_safety_scoped"].items():
            lines.append(
                "| "
                f"{evaluation} | "
                f"{scope.get('raw_post_rows', 0)} | "
                f"{scope.get('distinct_communities', 0)} | "
                f"{scope.get('safety_eligible_communities', 0)} | "
                f"{scope.get('safety_ineligible_communities', 0)} | "
                f"{scope.get('eligible_communities_with_relevance_judgment', 0)}/"
                f"{scope.get('safety_eligible_communities', 0)} | "
                f"{scope.get('eligible_relevance_distribution', {})} | N/A |"
            )
        lines.append("")

    # ------------------------------------------------
    # FALLBACK FOR OLDER RESULT STRUCTURE
    # ------------------------------------------------

    if (
        "communities" not in results
        and "threads" not in results
    ):

        for case_id, case in results.items():

            lines.extend(
                [
                    f"## {case_id}",
                    "",
                ]
            )

            if "community_metrics" in case:

                lines.extend(
                    [
                        "### Community Retrieval",
                        "",
                    ]
                )

                _write_source_table(
                    lines,
                    case["community_metrics"],
                )

            if "thread_metrics" in case:

                lines.extend(
                    [
                        "### Thread Retrieval",
                        "",
                    ]
                )

                _write_thread_table(
                    lines,
                    case["thread_metrics"],
                )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
