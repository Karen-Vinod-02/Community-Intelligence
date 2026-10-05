"""Reproduce the scoped community and thread evaluation audit.

This script is read-only with respect to source data. It writes a dated JSON
and Markdown report under results/ and never treats a safety exclusion as a
relevance judgment.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from evaluation.metrics import ndcg_at_k, reciprocal_rank


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "results" / "evaluation_scope_audit_20261004"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ranked_metrics(items: list[dict], item_type: str, annotations: dict):
    """Measure a supplied ranked list only when every unique item is judged."""
    id_key = "post_id" if item_type == "thread" else "subreddit"
    ids = [str(item.get(id_key) or item.get("id") or "").strip().lower() for item in items]
    ids = [item.removeprefix("r/") for item in ids if item]
    unique_ids = list(dict.fromkeys(ids))
    labels = [annotations.get((item_type, item_id)) for item_id in unique_ids]
    judged = [label for label in labels if label is not None]
    complete = bool(unique_ids) and len(judged) == len(unique_ids)
    return {
        "input_rows": len(items),
        "unique_ranked_items": len(unique_ids),
        "duplicate_rows_removed_for_metric": len(ids) - len(unique_ids),
        "annotated_items": len(judged),
        "unannotated_items": len(unique_ids) - len(judged),
        "judgments_complete": complete,
        "relevance_distribution": dict(sorted(Counter(judged).items())),
        "ndcg@10": ndcg_at_k(judged, 10) if complete else None,
        "mrr": reciprocal_rank(judged) if complete else None,
        "ranked_ids": unique_ids,
    }


def load_relevance_annotations(path: Path):
    rows = read_csv(path)
    annotations = {}
    for row in rows:
        raw = row.get("relevance", "").strip()
        if not raw:
            continue
        key = (row["item_type"].strip(), row["item_id"].strip().lower().removeprefix("r/"))
        if key in annotations:
            raise ValueError(f"Duplicate relevance annotation for {key}")
        value = int(raw)
        if value < 0 or value > 3:
            raise ValueError(f"Invalid relevance value {value} for {key}")
        annotations[key] = value
    return annotations, rows


def build_audit():
    paths = {
        "active_annotations": DATA / "annotations" / "latest" / "annotations.csv",
        "safety_screen": DATA / "pools" / "latest" / "agent_reach_safety_screen.csv",
        "mcp_ranked_communities": DATA / "normalized" / "latest" / "study_accountability" / "communities.json",
        "selected_threads": DATA / "normalized" / "latest" / "study_accountability" / "threads.json",
        "mcp_raw": DATA / "raw" / "latest" / "reddit_mcp" / "study_accountability.json",
        "agent_reach_raw": DATA / "raw" / "latest" / "agent_reach" / "study_accountability.json",
        "excluded_rows_archive_do_not_score": DATA / "annotations" / "latest" / "auto_screen_exclusions_not_relevance.csv",
        "case_configuration": ROOT / "cases" / "retrieval_cases.json",
        "pipeline_implementation": ROOT / "run_experiment.py",
        "metric_implementation": ROOT / "evaluation" / "metrics.py",
        "evaluator_implementation": ROOT / "evaluate.py",
        "report_implementation": ROOT / "evaluation" / "report.py",
        "audit_implementation": ROOT / "audit_evaluation.py",
        "annotation_pool_builder": ROOT / "build_annotation_pool.py",
        "annotation_merge_script": ROOT / "merge_annotations.py",
        "benchmark_summary": ROOT.parent / "benchmark" / "results" / "comparison_summary.json",
        "benchmark_pullpush_raw": ROOT.parent / "benchmark" / "results" / "pullpush_discovery_raw.json",
        "benchmark_arctic_posts_raw": ROOT.parent / "benchmark" / "results" / "arcticshift_posts_raw.json",
        "benchmark_arctic_comments_raw": ROOT.parent / "benchmark" / "results" / "arcticshift_comments_raw.json",
    }
    def relative_source_path(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path.relative_to(ROOT.parent))

    annotations, annotation_rows = load_relevance_annotations(paths["active_annotations"])
    safety_rows = read_csv(paths["safety_screen"])
    safety = {row["item_id"].strip().lower(): row["include"].strip().lower() == "yes" for row in safety_rows}

    communities = read_json(paths["mcp_ranked_communities"])
    threads_data = read_json(paths["selected_threads"])
    mcp_raw = read_json(paths["mcp_raw"])
    agent_raw = read_json(paths["agent_reach_raw"])

    mcp_ranked = communities["ranked"]
    mcp_metrics = ranked_metrics(mcp_ranked, "community", annotations)

    # Agent-Reach rows are post hits, with query-local post rank. There is no
    # global ranked community list in these data, so only report safety and
    # judgment coverage for the eligible unique communities.
    agent_communities = list(dict.fromkeys(
        str(row.get("subreddit", "")).strip().lower().removeprefix("r/")
        for row in agent_raw
        if row.get("subreddit")
    ))
    agent_safe = [name for name in agent_communities if safety.get(name) is True]
    agent_excluded = [name for name in agent_communities if safety.get(name) is False]
    agent_coverage = {
        "raw_post_rows": len(agent_raw),
        "distinct_communities": len(agent_communities),
        "distinct_communities_with_safety_decision": sum(name in safety for name in agent_communities),
        "safety_eligible_communities": len(agent_safe),
        "safety_ineligible_communities": len(agent_excluded),
        "eligible_communities_with_relevance_judgment": sum(("community", name) in annotations for name in agent_safe),
        "eligible_relevance_distribution": dict(sorted(Counter(
            annotations[("community", name)] for name in agent_safe if ("community", name) in annotations
        ).items())),
        "rank_metrics": None,
        "reason_rank_metrics_unavailable": "Raw rows are post hits ranked within five separate queries; no global ranked community list exists. Safety eligibility does not supply ranking order.",
        "eligible_ids": agent_safe,
        "ineligible_ids": agent_excluded,
    }

    thread_items = threads_data["selected"]
    thread_metrics = ranked_metrics(thread_items, "thread", annotations)
    ranked_thread_rows = [
        {
            "rank": rank,
            "post_id": str(item.get("post_id") or item.get("id") or ""),
            "subreddit": item.get("subreddit", ""),
            "title": item.get("title", ""),
            "score": item.get("score", 0),
            "num_comments": item.get("num_comments", 0),
            "relevance": annotations.get(("thread", str(item.get("post_id") or item.get("id") or "").lower())),
        }
        for rank, item in enumerate(thread_items, start=1)
    ]
    thread_annotation_by_id = {
        row["item_id"].strip().lower(): row
        for row in annotation_rows
        if row["item_type"].strip() == "thread" and row.get("relevance", "").strip()
    }
    thread_evidence = {
        "selected_threads": len(thread_items),
        "threads_with_comments": sum(bool(item.get("comments")) for item in thread_items),
        "comments_retrieved_total": sum(len(item.get("comments", [])) for item in thread_items),
        "threads_with_title_and_text": sum(bool(item.get("title", "").strip()) and bool(item.get("text", "").strip()) for item in thread_items),
        "threads_with_url": sum(bool(item.get("url")) for item in thread_items),
        "relevance_notes_present": sum(bool(thread_annotation_by_id.get(str(item.get("post_id", "")).lower(), {}).get("evidence", "").strip()) for item in thread_items),
        "retrieval_sources": dict(Counter(item.get("source", "unknown") for item in thread_items)),
        "discovery_sources": dict(Counter(source for item in thread_items for source in item.get("discovery_sources", []))),
    }

    # Verify pipeline order as stored: deduplicate_threads orders by
    # (num_comments, score), descending, then top_threads truncates to 15.
    sort_keys = [(int(item.get("num_comments", 0) or 0), float(item.get("score", 0) or 0)) for item in thread_items]
    expected = sorted(sort_keys, reverse=True)
    duplicate_thread_ids = len({item.get("post_id") for item in thread_items}) != len(thread_items)

    report = {
        "generated_on": "2026-10-04",
        "scope": "Existing study_accountability artifacts only; no new annotation or safety decisions.",
        "sources": {name: {"path": relative_source_path(path), "sha256": sha256(path)} for name, path in paths.items()},
        "community_evaluation": {
            "mcp_pipeline_ranked_list": mcp_metrics,
            "agent_reach_safety_scoped_findings": agent_coverage,
            "head_to_head_ranking_comparison": None,
            "head_to_head_limitation": "The MCP artifact is a ranked community list; Agent-Reach artifacts are query-local post hits. The common judged, safety-eligible candidate intersection is too small to establish a meaningful paired ranking comparison.",
            "excluded_annotation_archive_rows": len(read_csv(paths["excluded_rows_archive_do_not_score"])),
            "excluded_annotation_archive_policy": "Not loaded as relevance labels; its archived zeroes must not be interpreted as relevance judgments.",
        },
        "thread_evaluation": {
            "ranking_file": "data/normalized/latest/study_accountability/threads.json:selected",
            "ranking_order_verified": sort_keys == expected,
            "duplicate_post_ids": duplicate_thread_ids,
            "selection_limit": 15,
            "retrieved_candidate_count_available": None,
            "retrieved_candidate_count_limitation": "The saved normalized file contains only selected threads; the complete pre-truncation unique candidate pool is not retained here.",
            "metrics": thread_metrics,
            "ranked_items": ranked_thread_rows,
            "evidence_availability": thread_evidence,
            "interpretation": "nDCG@10 and MRR describe ordering quality within the selected, fully judged 15-item list. They do not measure retrieval recall or candidate-pool coverage. All labels are positive (2 or 3), so MRR=1.0 is guaranteed by a relevant first result and relevance discrimination is limited.",
            "metric_definitions": "nDCG uses gain 2^relevance-1 and logarithmic discount; the ideal list is sorted from all judged unique selected items. MRR returns the reciprocal rank of the first item with relevance > 0.",
            "actual_selection_order": "Deduplicate globally by post_id, sort descending by (num_comments, score), then take the first 15. Comments are fetched after this ordering and do not rerank the selected list.",
            "production_relevance_ranker_used": False,
        },
        "whole_system_experiments": {
            "pullpush_arctic_shift_benchmark": "Standalone API reliability/latency and response-volume benchmark; no relevance labels. PullPush 0/5 HTTP 502; Arctic Shift posts 5/5 HTTP 200 at 50 results/call (mean 1.145 s); comments 5/5 HTTP 200 but only 3 comments total (mean 1.102 s); official Reddit discovery was skipped for missing credentials. Discovery had no candidates, so retrieval used five configured known subreddits. Raw post samples do not retain timestamps for all returned records, so the 90-day window cannot be independently validated. Not a relevance comparison.",
            "community_discovery": "Reddit MCP query-level discovery and confidence/query-coverage baseline, with Agent-Reach supplementary post search. Arctic Shift is subreddit-scoped enrichment.",
            "thread_retrieval": "Arctic Shift retrieves threads/comments from top five MCP-ranked communities; selected-list annotation supports Evaluation C.",
            "backend_ranking": "Production backend includes community filtering/ranking and retrieval services; the experiment's saved ranked lists are the evidence assessed here. Backend unit tests cover problem representation and subreddit integrity.",
        },
        "pipeline_trace": [
            "Raw Reddit MCP and Agent-Reach results are saved separately. MCP results are deduplicated by normalized subreddit, ranked from confidence/query coverage/source rank, and selected for subreddit-scoped retrieval. Agent-Reach is supplementary and does not enter that ranked MCP pool.",
            "Arctic Shift queries each selected community. Thread candidates are deduplicated by post_id, ordered descending by num_comments then score, and truncated to the configured top_threads=15 before comments are fetched.",
            "build_annotation_pool.py exports selected communities and threads to the pool CSV; relevance annotations are merged/stored separately. The Agent-Reach safety CSV is an eligibility screen, separate from the active relevance annotation file.",
        ],
    }
    return report


def render_markdown(report: dict) -> str:
    community = report["community_evaluation"]
    mcp = community["mcp_pipeline_ranked_list"]
    agent = community["agent_reach_safety_scoped_findings"]
    thread = report["thread_evaluation"]
    metrics = thread["metrics"]
    evidence = thread["evidence_availability"]
    ranking_table = "\n".join(
        "| {rank} | `{post_id}` | r/{subreddit} | {num_comments} | {score} | {relevance} | {title} |".format(
            **{**item, "title": item["title"].replace("|", "\\|")}
        )
        for item in thread["ranked_items"]
    )
    return f"""# Evaluation Scope Audit — 2026-10-04

## Scope and method

This report is reproduced from the existing artifacts listed with SHA-256 hashes in the adjacent JSON file. It does not add judgments. Safety eligibility and relevance remain separate. The archived `auto_screen_exclusions_not_relevance.csv` is deliberately not loaded as relevance data; it contains {community['excluded_annotation_archive_rows']} rows with zeroes that are not valid relevance judgments.

## Experiment inventory

| Experiment | Question and method | Evidence and limitation |
|---|---|---|
| PullPush vs Arctic Shift benchmark | Can Reddit-wide keyword discovery and subreddit-scoped post/comment retrieval run, within a recent time window and at what latency? | Existing run: PullPush 0/5 (all HTTP 502; no successful-call latency); Arctic Shift posts 5/5 HTTP 200, 250 results, mean 1.145 s and p95 1.317 s; comments 5/5 HTTP 200, 3 total (0.6/call), mean 1.102 s and p95 1.219 s. With no discovery candidates, posts used the five configured known subreddits. Operational availability/volume only, no relevance judgments. |
| Community discovery | Compare MCP ranked subreddit candidates and inspect Agent-Reach supplementary discovery. | MCP pipeline ranked list: {mcp['unique_ranked_items']} unique items, {mcp['annotated_items']}/{mcp['unique_ranked_items']} judged, nDCG@10 {mcp['ndcg@10']:.4f}, MRR {mcp['mrr']:.4f}. Agent-Reach raw rows are post hits: {agent['raw_post_rows']} rows, {agent['distinct_communities']} distinct communities; {agent['safety_eligible_communities']} safety eligible and {agent['safety_ineligible_communities']} ineligible. All eligible communities have relevance labels, distribution {agent['eligible_relevance_distribution']}; no ranking metric is valid because there is no global ranked community list. |
| Thread retrieval (Evaluation C) | Retrieve posts/comments from selected MCP communities, then score the saved selected ranking against human relevance judgments. | {metrics['annotated_items']}/{metrics['unique_ranked_items']} selected unique threads judged. This is selected-list ranking quality, not retrieval recall or candidate-pool coverage. |
| Backend ranking/services | Production service has filtering, community ranking, discovery and Reddit retrieval components. | Backend tests exist for problem representation and subreddit integrity. The saved experiment artifacts, rather than production-run logs, are the measured ranking evidence here. |

## Data and ranking trace

Raw MCP and Agent-Reach results are saved separately. MCP candidates are deduplicated by subreddit, then ranked using confidence, distinct-query coverage, best source rank, and subreddit tie-break. Only MCP enters the normalized community retrieval pipeline; Agent-Reach stays supplementary. Arctic Shift retrieves posts from the selected five MCP communities. Threads are deduplicated by post ID, sorted by comment count and post score, truncated to 15, then comments are fetched. The annotation-pool builder combines those selected communities and threads; safety eligibility and relevance judgments are stored in separate CSV artifacts.

## Community comparison scope

There is no fair head-to-head ranking comparison. MCP has a ranked community list. Agent-Reach has 75 post hits across five query-local rankings, collapsing to 32 communities, with no normalized global community ranking. Safety screening leaves 13 eligible communities; screening decisions do not imply relevance labels. `getstudying` is safety-ineligible for Agent-Reach while separately carrying an MCP community relevance judgment; that judgment stays attached to the community and does not override Agent-Reach safety eligibility. Report MCP ranked-list metrics and Agent-Reach safety-scoped coverage separately.

## Evaluation C: exact saved thread ranking

The selected list has {metrics['unique_ranked_items']} unique post IDs, duplicate IDs: {thread['duplicate_post_ids']}. Its order is verified against the pipeline's descending `(num_comments, score)` sort and 15-item truncation. Every selected item has a relevance judgment, so nDCG@10 and MRR are computable for this list.

| Measure | Result |
|---|---:|
| Judged coverage | {metrics['annotated_items']}/{metrics['unique_ranked_items']} |
| nDCG@10 | {metrics['ndcg@10']:.4f} |
| MRR | {metrics['mrr']:.4f} |
| Labels | {metrics['relevance_distribution']} |
| Threads with retrieved comments | {evidence['threads_with_comments']}/{evidence['selected_threads']} |
| Retrieved comments represented in saved artifacts | {evidence['comments_retrieved_total']} |
| Retrieval source / discovery provenance | {evidence['retrieval_sources']} / {evidence['discovery_sources']} |
| Relevance notes present | {evidence['relevance_notes_present']}/{metrics['unique_ranked_items']} |

| Rank | Post ID | Community | Reddit comment count | Post score | Relevance | Title |
|---:|---|---|---:|---:|---:|---|
{ranking_table}

All labels are positive (2 or 3), so MRR=1.0 follows from the first ranked item being relevant. nDCG@10 measures graded ordering among the selected 15. The pre-truncation unique candidate count is not saved, therefore retrieval coverage/recall cannot be calculated. Ranking uses comment count and post score; comments are fetched after selection. The separate `evaluation/rank.py` relevance-oriented scoring function does not produce this saved ranking.

## Benchmark interpretation

The benchmark separates endpoint success/latency/volume from retrieval relevance. PullPush had five 502 failures and therefore no successful-call latency summary. Arctic Shift returned successful HTTP responses for the configured subreddit checks: post requests averaged 1.145 seconds (p95 1.317), and comment requests averaged 1.102 seconds (p95 1.219). Those five comment calls returned only three comments total; HTTP success is not the same as useful evidence coverage. The saved samples do not preserve timestamps for all 250 returned posts, so actual compliance with the 90-day window cannot be independently checked. These data do not show a relevance advantage for either source.

## Verified limitations and bugs

- **Method limitation:** no paired global ranked community lists for MCP and Agent-Reach; source populations and ranking units differ.
- **Method limitation:** safety exclusions are eligibility decisions, not relevance zeroes. The archived exclusion CSV must not enter the relevance evaluator.
- **Method limitation:** thread evaluation covers only the top 15 selected items; the full unique candidate pool was not persisted.
- **Method limitation:** Evaluation C measures a popularity/engagement ordering (`num_comments`, then score), not the relevance-oriented `rank_threads` score.
- **Evidence limitation:** benchmark success counts can hide zero-result calls; the comments endpoint had 5/5 HTTP success but 3 comments total.
- **Verified code/report mismatch:** README describes a top-30 thread selection, but case configuration and pipeline select 15.
- **Verified implementation mismatch:** `deduplicate_threads` documentation says subreddit + post ID, while implementation deduplicates by post ID alone. Reddit post IDs are globally unique, so this does not alter the present list.
- **Evaluator correction:** raw MCP query rankings and Agent-Reach post-hit rankings are no longer flattened into community ranking metrics. The evaluator reports only the normalized MCP community ranking and emits Agent-Reach safety-scoped coverage separately. Raw data and archived exclusion labels remain untouched.

## Reproduction

From this directory run `python audit_evaluation.py`. The script reads the hashed inputs above and writes a JSON and Markdown report under `results/evaluation_scope_audit_20261004/`.

## Prioritized next steps

1. Preserve the full pre-truncation deduplicated thread pool in future runs so retrieval coverage can be reported separately from ranking quality.
2. Keep Agent-Reach safety decisions in an eligibility field and keep excluded rows out of relevance metrics; use query-level post retrieval measurements unless a true ranked community output exists.
3. Re-run the standalone API benchmark when source availability needs a current operational measurement; retain HTTP status, result count, latency, and date-window validation as separate measures.
4. Evaluate backend community and thread ranking against the same case artifacts only after its actual ranked output is captured; do not treat API reliability as relevance.
"""


def main():
    audit = build_audit()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT / "report.md").write_text(render_markdown(audit), encoding="utf-8")
    print(f"Audit written to {OUTPUT}")


if __name__ == "__main__":
    main()
