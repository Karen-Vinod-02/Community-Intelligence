
import csv
import argparse
import json
from pathlib import Path

from evaluation.metrics import (
    ndcg_at_k,
    reciprocal_rank,
)

from evaluation.report import (
    save_json,
    save_report,
)


ROOT = Path(__file__).resolve().parent

ANNOTATIONS = (
    ROOT
    / "data"
    / "annotations"
    / "latest"
    / "annotations.csv"
)

RAW_DIR = (
    ROOT
    / "data"
    / "raw"
    / "latest"
)

NORMALIZED_DIR = (
    ROOT
    / "data"
    / "normalized"
    / "latest"
)

SAFETY_SCREEN = (
    ROOT
    / "data"
    / "pools"
    / "latest"
    / "agent_reach_safety_screen.csv"
)

RESULTS_DIR = (
    ROOT
    / "results"
    / "latest"
)


# ------------------------------------------------
# LOAD ANNOTATIONS
# ------------------------------------------------

def load_annotations():
    """
    Load relevance annotations.

    Expected CSV columns:
        item_type
        item_id
        relevance
        evidence
        notes

    Relevance must be an integer from 0 to 3.
    Evidence and notes remain textual.
    """

    if not ANNOTATIONS.exists():
        raise FileNotFoundError(
            f"Annotation file not found:\n{ANNOTATIONS}"
        )

    annotations = {}

    with ANNOTATIONS.open(
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                "Annotation file has no header."
            )

        required_columns = {
            "item_type",
            "item_id",
            "relevance",
        }

        missing = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing:
            raise ValueError(
                "Annotation file is missing required "
                f"columns: {sorted(missing)}"
            )

        row_count = 0

        for row in reader:

            row_count += 1

            item_type = (
                row["item_type"].strip()
            )

            item_id = (
                row["item_id"].strip()
            )

            relevance_raw = (
                row["relevance"].strip()
            )

            if not item_type or not item_id:
                raise ValueError(
                    f"Missing item_type or item_id "
                    f"at CSV row {row_count + 1}."
                )

            if not relevance_raw:
                raise ValueError(
                    f"Missing relevance for "
                    f"{item_type}/{item_id}"
                )

            try:
                relevance = int(
                    relevance_raw
                )
            except ValueError:
                raise ValueError(
                    f"Invalid relevance value "
                    f"'{relevance_raw}' for "
                    f"{item_type}/{item_id}. "
                    "Expected an integer from 0 to 3."
                )

            if not 0 <= relevance <= 3:
                raise ValueError(
                    f"Relevance for "
                    f"{item_type}/{item_id} is "
                    f"{relevance}. Expected 0-3."
                )

            key = (
                item_type,
                item_id,
            )

            if key in annotations:
                raise ValueError(
                    f"Duplicate annotation for "
                    f"{item_type}/{item_id}"
                )

            annotations[key] = {
                "relevance": relevance,
                "evidence": (
                    row.get("evidence", "").strip()
                ),
                "notes": (
                    row.get("notes", "").strip()
                ),
            }

    if row_count == 0:
        raise ValueError(
            f"{ANNOTATIONS} is empty.\n\n"
            "Save the completed annotations into "
            "the active annotation pool first."
        )

    return annotations


# ------------------------------------------------
# ITEM ID HELPERS
# ------------------------------------------------

def get_item_id(
    item,
    item_type,
):
    """
    Extract the annotation ID from a raw or
    normalized item.
    """

    if item_type == "community":
        return (
            item.get("subreddit")
            or item.get("name")
            or item.get("community")
        )

    if item_type == "thread":
        return (
            item.get("post_id")
            or item.get("id")
        )

    raise ValueError(
        f"Unsupported item type: {item_type}"
    )


# ------------------------------------------------
# EVALUATE ONE RANKED LIST
# ------------------------------------------------

def evaluate_source(
    items,
    item_type,
    annotations,
):
    """
    Evaluate an already-ranked list.

    Original ranking order is preserved.

    Unannotated items are counted but are NOT
    assigned relevance 0.

    nDCG@10 and MRR are calculated only when
    every item in the evaluated list is annotated.

    This avoids evaluating incomplete ground truth
    as if missing judgments meant irrelevance.
    """

    annotated_count = 0
    unannotated_count = 0
    seen_ids = set()

    labels = []

    for item in items:

        item_id = get_item_id(
            item,
            item_type,
        )

        if not item_id:
            continue

        item_id = str(item_id).strip()

        if not item_id:
            continue

        # Avoid counting the same item more than
        # once in a single evaluated ranking.
        if item_id in seen_ids:
            continue

        seen_ids.add(item_id)

        annotation = annotations.get(
            (
                item_type,
                item_id,
            )
        )

        if annotation is None:

            unannotated_count += 1

        else:

            annotated_count += 1

            labels.append(
                annotation["relevance"]
            )

    total_items = (
        annotated_count
        + unannotated_count
    )

    # Metrics are calculated only if all retrieved
    # items have judgments and the list is nonempty.
    complete_judgments = (
        total_items > 0
        and unannotated_count == 0
    )

    if complete_judgments:

        ndcg = ndcg_at_k(
            labels,
            10,
        )

        mrr = reciprocal_rank(
            labels
        )

    else:

        ndcg = None
        mrr = None

    return {
        "n_items": total_items,
        "annotated_items": annotated_count,
        "unannotated_items": unannotated_count,
        "judgments_complete": complete_judgments,
        "ndcg@10": ndcg,
        "mrr": mrr,
    }


# ------------------------------------------------
# EVALUATE RAW COMMUNITY SOURCE
# ------------------------------------------------

def evaluate_raw_community_source(
    source,
    annotations,
):
    """
    Evaluate raw community discovery results.

    Supports:
        [...]
        {"data": [...]}
        {"subreddits": [...]}
        {"communities": [...]}
    """

    source_dir = (
        RAW_DIR
        / source
    )

    if not source_dir.exists():
        return {}

    source_results = {}

    for path in sorted(
        source_dir.glob("*.json")
    ):

        if path.name.endswith(
            "_error.json"
        ):
            continue

        case_id = path.stem

        try:
            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

        except (
            json.JSONDecodeError,
            OSError,
        ) as exc:

            print(
                f"Warning: could not read "
                f"{path}: {exc}"
            )

            continue

        if isinstance(data, list):

            items = data

        elif isinstance(data, dict):

            items = None

            for key in (
                "subreddits",
                "data",
                "communities",
            ):

                if isinstance(
                    data.get(key),
                    list,
                ):

                    items = data[key]
                    break

            if items is None:

                print(
                    f"Warning: no community list "
                    f"found in {path}"
                )

                continue

        else:

            print(
                f"Warning: unexpected data format "
                f"in {path}"
            )

            continue

        source_results[case_id] = (
            evaluate_source(
                items,
                "community",
                annotations,
            )
        )

    return source_results


# ------------------------------------------------
# COMMUNITY EVALUATION
# ------------------------------------------------

def evaluate_communities(
    annotations,
):
    """
    Evaluate only the normalized MCP-ranked list.

    Important:
        - Raw source records contain query-local rankings;
          flattening them would invent a global order.
        - The normalized baseline uses the actual
          pipeline-generated 'ranked' list.
    """

    results = {}

    # --------------------------------------------
    # MCP RANKED BASELINE
    # --------------------------------------------

    mcp_ranked_baseline = {}

    if NORMALIZED_DIR.exists():

        for case_dir in sorted(
            NORMALIZED_DIR.iterdir()
        ):

            if not case_dir.is_dir():
                continue

            path = (
                case_dir
                / "communities.json"
            )

            if not path.exists():
                continue

            try:
                data = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )

            except (
                json.JSONDecodeError,
                OSError,
            ) as exc:

                print(
                    f"Warning: could not read "
                    f"{path}: {exc}"
                )

                continue

            if not isinstance(data, dict):

                print(
                    f"Warning: unexpected "
                    f"communities.json format: "
                    f"{path}"
                )

                continue

            # Evaluate the actual pipeline ranking.
            # Do not rerank the deduplicated pool here.
            ranked = data.get(
                "ranked",
                [],
            )

            if not isinstance(
                ranked,
                list,
            ):

                print(
                    f"Warning: ranked communities "
                    f"is not a list in {path}"
                )

                continue

            mcp_ranked_baseline[
                case_dir.name
            ] = evaluate_source(
                ranked,
                "community",
                annotations,
            )

    if mcp_ranked_baseline:

        results[
            "mcp_ranked_baseline"
        ] = mcp_ranked_baseline

    return results


def evaluate_agent_reach_safety_scope(annotations):
    """Summarize screened Agent-Reach communities without ranking metrics."""
    raw_dir = RAW_DIR / "agent_reach"
    if not SAFETY_SCREEN.exists() or not raw_dir.exists():
        return {}

    safety = {}
    with SAFETY_SCREEN.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            item_id = row.get("item_id", "").strip().lower().removeprefix("r/")
            if item_id:
                safety[item_id] = row.get("include", "").strip().lower() == "yes"

    output = {}
    for path in sorted(raw_dir.glob("*.json")):
        if path.name.endswith("_error.json"):
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(rows, list):
            continue

        communities = list(dict.fromkeys(
            str(row.get("subreddit") or row.get("name") or row.get("community") or "")
            .strip().lower().removeprefix("r/")
            for row in rows
        ))
        communities = [name for name in communities if name]
        eligible = [name for name in communities if safety.get(name) is True]
        ineligible = [name for name in communities if safety.get(name) is False]
        unreviewed = [name for name in communities if name not in safety]
        judged = [
            annotations[("community", name)]["relevance"]
            for name in eligible
            if ("community", name) in annotations
        ]
        output[path.stem] = {
            "raw_post_rows": len(rows),
            "distinct_communities": len(communities),
            "safety_eligible_communities": len(eligible),
            "safety_ineligible_communities": len(ineligible),
            "communities_without_safety_decision": len(unreviewed),
            "eligible_communities_with_relevance_judgment": len(judged),
            "eligible_communities_without_relevance_judgment": len(eligible) - len(judged),
            "eligible_relevance_distribution": {
                str(label): judged.count(label)
                for label in sorted(set(judged))
            },
            "ranking_metrics": None,
            "ranking_limitation": (
                "Raw rows are post hits ranked within separate queries; "
                "no global ranked community list exists."
            ),
        }
    return output


# ------------------------------------------------
# THREAD EVALUATION
# ------------------------------------------------

def evaluate_threads(
    annotations,
):
    """
    Evaluate the normalized thread ranking.

    Supports:
        [...]
        {"selected": [...]}
        {"ranked": [...]}
        {"threads": [...]}
        {"all_candidates": [...]}

    The first matching structure is used.
    """

    results = {}

    if not NORMALIZED_DIR.exists():
        return results

    for case_dir in sorted(
        NORMALIZED_DIR.iterdir()
    ):

        if not case_dir.is_dir():
            continue

        path = (
            case_dir
            / "threads.json"
        )

        if not path.exists():
            continue

        try:
            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

        except (
            json.JSONDecodeError,
            OSError,
        ) as exc:

            print(
                f"Warning: could not read "
                f"{path}: {exc}"
            )

            continue

        if isinstance(data, list):

            threads = data

        elif isinstance(data, dict):

            threads = None

            for key in (
                "selected",
                "ranked",
                "threads",
                "all_candidates",
            ):

                if isinstance(
                    data.get(key),
                    list,
                ):

                    threads = data[key]
                    break

            if threads is None:

                print(
                    f"Warning: could not find "
                    f"thread list in {path}"
                )

                continue

        else:

            print(
                f"Warning: unexpected thread "
                f"data format in {path}"
            )

            continue

        results[
            case_dir.name
        ] = evaluate_source(
            threads,
            "thread",
            annotations,
        )

    return results


# ------------------------------------------------
# PRINT SUMMARY
# ------------------------------------------------

def format_metric(value):
    """
    Format a metric without failing on None.
    """

    if value is None:
        return "N/A"

    return f"{value:.4f}"


def print_metric_object(
    label,
    metrics,
    indent="  ",
):
    print(
        f"{indent}{label}:"
    )

    print(
        f"{indent}  n_items: "
        f"{metrics['n_items']}"
    )

    print(
        f"{indent}  annotated: "
        f"{metrics['annotated_items']}"
    )

    print(
        f"{indent}  unannotated: "
        f"{metrics['unannotated_items']}"
    )

    print(
        f"{indent}  judgments_complete: "
        f"{metrics['judgments_complete']}"
    )

    print(
        f"{indent}  nDCG@10: "
        f"{format_metric(metrics['ndcg@10'])}"
    )

    print(
        f"{indent}  MRR: "
        f"{format_metric(metrics['mrr'])}"
    )


def print_summary(results):

    print()
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for category, category_results in (
        results.items()
    ):

        print()
        print(category.upper())

        if not category_results:

            print("  No results.")
            continue

        if category == "community_scope":
            for scope_name, cases in category_results.items():
                print(f"  {scope_name}:")
                for case_id, scope in cases.items():
                    print(
                        f"    {case_id}: "
                        f"{scope['safety_eligible_communities']} safety eligible / "
                        f"{scope['distinct_communities']} distinct; "
                        f"{scope['eligible_communities_with_relevance_judgment']} "
                        "eligible communities judged; ranking metrics N/A"
                    )
            continue

        for name, metrics in (
            category_results.items()
        ):

            # A direct metric object
            if (
                isinstance(metrics, dict)
                and "ndcg@10" in metrics
            ):

                print_metric_object(
                    name,
                    metrics,
                )

            # Source -> case -> metrics
            elif isinstance(metrics, dict):

                print(f"  {name}:")

                if not metrics:
                    print("    No results.")
                    continue

                for case_id, case_metrics in (
                    metrics.items()
                ):

                    if (
                        isinstance(
                            case_metrics,
                            dict,
                        )
                        and "ndcg@10" in case_metrics
                    ):

                        print_metric_object(
                            case_id,
                            case_metrics,
                            indent="    ",
                        )


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main(output_dir=RESULTS_DIR):

    print(
        "Loading annotations..."
    )

    annotations = load_annotations()

    print(
        f"Loaded {len(annotations)} annotations."
    )

    community_results = (
        evaluate_communities(
            annotations
        )
    )

    agent_reach_scope = evaluate_agent_reach_safety_scope(
        annotations
    )

    thread_results = (
        evaluate_threads(
            annotations
        )
    )

    results = {
        "communities": community_results,
        "community_scope": {
            "agent_reach_safety_scoped": agent_reach_scope,
        },
        "threads": thread_results,
    }

    # --------------------------------------------
    # SAVE
    # --------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        output_dir / "metrics.json",
        results,
    )

    save_report(
        output_dir / "report.md",
        results,
    )

    print_summary(results)

    print()
    print("Evaluation complete.")

    print(
        f"Metrics: "
        f"{output_dir / 'metrics.json'}"
    )

    print(
        f"Report: "
        f"{output_dir / 'report.md'}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved retrieval artifacts.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Directory for metrics/report output (default: {RESULTS_DIR})",
    )
    args = parser.parse_args()
    main(args.output_dir)
