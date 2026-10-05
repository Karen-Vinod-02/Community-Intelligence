"""Write the bounded production-ranker evaluation status and input hashes."""

from __future__ import annotations

import hashlib
import csv
import json
import os
import subprocess
from pathlib import Path

from evaluation.metrics import ndcg_at_k, reciprocal_rank


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OUTPUT = ROOT / "results" / "production_ranker_evaluation_20261004"
TRACE_PATH = (
    Path(os.environ.get("TEMP", Path.home() / "AppData/Local/Temp"))
    / "commint-evaluation-traces-20261004"
    / "98bc799137b24ba794666881f72854b8.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_report() -> dict:
    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    trace_summary = {
        "trace_id": trace.get("trace_id"),
        "captured_at": trace.get("captured_at"),
        "providers": trace.get("providers"),
        "query_count": len(trace.get("queries", [])),
        "discovery_query_count": len(trace.get("discovery", [])),
        "discovery_candidate_counts": [
            len(item.get("subreddits", [])) for item in trace.get("discovery", [])
        ],
        "discovered_community_count": len(trace.get("discovered_communities", [])),
        "bounded_community_count": len(trace.get("bounded_communities", [])),
        "retrieval_attempt_count": len(trace.get("retrieval", [])),
        "community_filter_score_count": len(trace.get("community_filter", {}).get("scores", [])),
        "community_ranker_diagnostic_count": len(trace.get("community_ranker", {}).get("candidate_diagnostics", [])),
        "ranked_community_count": len(trace.get("ranked_output", [])),
        "returned_community_count": len(trace.get("final_response", {}).get("communities", [])),
        "trace_id_returned_in_response": trace.get("final_response", {}).get("evaluation_trace_id") == trace.get("trace_id"),
        "pipeline_error": trace.get("pipeline_error"),
    }
    inputs = {
        "active_annotations": ROOT / "data" / "annotations" / "latest" / "annotations.csv",
        "saved_communities": ROOT / "data" / "normalized" / "latest" / "study_accountability" / "communities.json",
        "saved_selected_threads": ROOT / "data" / "normalized" / "latest" / "study_accountability" / "threads.json",
        "existing_experimental_comparison": ROOT / "results" / "ranker_comparison_20261004" / "comparison.json",
        "production_ranker": REPO / "backend" / "app" / "services" / "ranking.py",
        "production_filter": REPO / "backend" / "app" / "services" / "community_filter.py",
        "community_service": REPO / "backend" / "app" / "services" / "community.py",
        "trace_writer": REPO / "backend" / "app" / "services" / "evaluation_trace.py",
        "production_api": REPO / "backend" / "app" / "api" / "analysis.py",
        "api_models": REPO / "backend" / "app" / "models" / "analysis.py",
    }
    annotations = {}
    with inputs["active_annotations"].open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("item_type") == "thread" and row.get("relevance", "").strip():
                annotations[row["item_id"].strip().lower()] = int(row["relevance"])
    saved_comparison = json.loads(
        inputs["existing_experimental_comparison"].read_text(encoding="utf-8-sig")
    )["thread_comparison"]
    baseline_ids = saved_comparison["baseline"]["ranked_ids"]
    heuristic_ids = saved_comparison["comment_engagement_reranker"]["ranked_ids"]
    if set(map(str.lower, baseline_ids)) != set(map(str.lower, heuristic_ids)):
        raise ValueError("Saved thread comparison does not use identical candidates")
    baseline_labels = [annotations[str(item).lower()] for item in baseline_ids]
    heuristic_labels = [annotations[str(item).lower()] for item in heuristic_ids]
    if ndcg_at_k(baseline_labels, 10) != saved_comparison["baseline"]["nDCG@10"]:
        raise ValueError("Recomputed baseline nDCG@10 differs from saved comparison")
    if ndcg_at_k(heuristic_labels, 10) != saved_comparison["comment_engagement_reranker"]["nDCG@10"]:
        raise ValueError("Recomputed heuristic nDCG@10 differs from saved comparison")
    if reciprocal_rank(baseline_labels) != saved_comparison["baseline"]["MRR"]:
        raise ValueError("Recomputed baseline MRR differs from saved comparison")
    if reciprocal_rank(heuristic_labels) != saved_comparison["comment_engagement_reranker"]["MRR"]:
        raise ValueError("Recomputed heuristic MRR differs from saved comparison")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    working_tree_dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()
    )
    return {
        "generated_on": "2026-10-04",
        "source_revision": revision,
        "working_tree_dirty": working_tree_dirty,
        "production_trace": {
            "status": "captured_empty_candidate_path",
            "attempts": 1,
            "request": "POST /api/analyze with capture_trace=true",
            "outcome": "The request produced a trace and final API response, but web discovery returned zero communities for every query. Retrieval, filter scoring, and community ranking were therefore not invoked. The trace validates the empty-candidate API path only, not candidate scoring instrumentation.",
            "client_observation": "The PowerShell HTTP client timed out at 240 seconds; a trace was present afterward and contains the completed final_response. The caller did not receive the HTTP response body.",
            "environment": "Sentence-transformer weights loaded. DDGS logged a Windows root-certificate-store access denial and upstream search timeouts; the API trace records empty discovery outputs but DDGS internally suppresses its provider errors.",
            "trace_path": f"%TEMP%/commint-evaluation-traces-20261004/{TRACE_PATH.name}",
            "trace_sha256": sha256(TRACE_PATH),
            "trace_summary": trace_summary,
        },
        "production_comparison": {
            "status": "not_valid_with_available_inputs",
            "reason": "The trace contains the exact request and output, but production discovery produced zero candidates. With no shared candidate communities there is no rank order to compare. Substituting the saved normalized threads would change the production candidate population; those artifacts contain only a popularity-selected top-15 list and lack the full pre-truncation pool.",
            "labels": "Production trace judged coverage is not applicable because the trace has zero candidate communities. Existing study_accountability labels apply to saved experiment artifacts and cannot provide candidate judgments for this empty production run.",
            "ndcg_at_10": None,
            "mrr": None,
        },
        "existing_same_candidate_diagnostic": {
            "status": "valid_for_saved_selected_thread_list_only",
            "ranker": "Standalone comment-engagement heuristic in experiments/commint-retrieval-experiment/evaluation/rank.py; this is not the production semantic community ranker.",
            "candidate_selection": "The existing selected top-15 normalized thread list, originally truncated after ordering by descending (num_comments, score). Both orders contain the same 15 IDs and use the same 15 labels.",
            "label_distribution": {"2": 4, "3": 11},
            "baseline": {"nDCG@10": 0.8356420858776415, "MRR": 1.0, "judged": 15, "candidates": 15},
            "comment_engagement": {"nDCG@10": 0.821686241587554, "MRR": 1.0, "judged": 15, "candidates": 15},
            "anonymous_baseline_label_order": baseline_labels,
            "anonymous_comment_engagement_label_order": heuristic_labels,
            "interpretation": "This small, selected-list diagnostic does not answer whether the production semantic ranker improves over the popularity baseline. All labels are positive, making MRR=1.0 uninformative for ordering differences; there is no retrieval coverage estimate.",
        },
        "verification": {
            "backend_tests": "10 tests run; 9 passed, 1 existing failure in test_problem_representation.py: parser returned 'small' while expected 'small businesses'.",
            "experiment_tests": "8 tests passed.",
            "git_diff_check": "passed; Git emitted only line-ending normalization warnings.",
            "trace_stays_outside_repository": True,
        },
        "input_hashes": {
            name: {"path": str(path.relative_to(REPO)), "sha256": sha256(path)}
            for name, path in inputs.items()
        },
        "commands": [
            ".\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v (run from backend)",
            ".\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v (run from experiment directory)",
            "git diff --check (repository root)",
            "python write_production_evaluation_status.py (experiment directory; writes only this report directory)",
        ],
    }


def render_markdown(report: dict) -> str:
    attempt = report["production_trace"]
    comparison = report["existing_same_candidate_diagnostic"]
    hashes = "\n".join(
        f"| `{value['path']}` | `{value['sha256']}` |"
        for value in report["input_hashes"].values()
    )
    trace = attempt["trace_summary"]
    return f"""# Production ranker evaluation status - {report['generated_on']}

## Outcome

The response-contract and opt-in tracing implementation has focused test coverage. One real `POST /api/analyze` request produced a trace with a completed final response, but discovery returned an **empty candidate path**, not a valid ranking evaluation. The PowerShell client timed out after 240 seconds and did not receive the response body. The trace remains in the OS temporary directory and is not copied into this repository. It includes the user description, so protect the local file.

{attempt['outcome']} {attempt['environment']}

Trace ID: `{trace['trace_id']}`; captured at `{trace['captured_at']}`; SHA-256: `{attempt['trace_sha256']}`. Providers: `{trace['providers']}`. The trace has {trace['query_count']} extracted queries, {trace['discovery_query_count']} discovery calls with candidate counts `{trace['discovery_candidate_counts']}`, {trace['retrieval_attempt_count']} retrieval attempts, {trace['community_filter_score_count']} filter scores, {trace['community_ranker_diagnostic_count']} community ranker diagnostics, {trace['ranked_community_count']} ranked results, and {trace['returned_community_count']} response communities. The trace ID was returned in the response: `{trace['trace_id_returned_in_response']}`. No description or post text is reproduced here.

## Production comparison

The production semantic community ranker was not scored because this production run discovered no candidate communities; retrieval, filtering, and ranking were skipped. There is no shared ranked candidate set or judged production item, so nDCG@10 and MRR are suppressed. The saved normalized artifacts contain a different, popularity-selected top-15 thread list without its complete pre-truncation pool. Reusing it would change the production candidate population and would not answer whether the production ranker improved on this request.

## Existing same-candidate diagnostic (not the production ranker)

The existing experiment compares the saved top-15 thread list with the standalone comment-engagement heuristic. Both have the same 15 candidates and all 15 judgments (labels: `{comparison['label_distribution']}`). Popularity baseline: nDCG@10 **{comparison['baseline']['nDCG@10']:.4f}**, MRR **{comparison['baseline']['MRR']:.4f}**. Comment-engagement heuristic: nDCG@10 **{comparison['comment_engagement']['nDCG@10']:.4f}**, MRR **{comparison['comment_engagement']['MRR']:.4f}**. This is a valid comparison only for that selected list and heuristic; it is not evidence about the production semantic ranker. All labels are positive, and no full pool exists to estimate retrieval coverage.

## Verification

- Backend suite: {report['verification']['backend_tests']}
- Experiment suite: {report['verification']['experiment_tests']}
- `git diff --check`: {report['verification']['git_diff_check']}

The unrelated parser failure is left unchanged. It concerns actor extraction for `A platform helping small businesses find affordable designers`, where the current value is `small` and the existing test expects `small businesses`.

## Reproduction and provenance

Run the commands listed in `evaluation_status.json`. The source revision and SHA-256 hashes below identify the code, active judgments, normalized candidate artifacts, and prior diagnostic comparison used for this report. The production trace is kept outside the repository at the path recorded in `evaluation_status.json`.

| Input | SHA-256 |
|---|---|
{hashes}

## Stop condition and next step

Do not claim that the production semantic ranker beats the popularity baseline from these results. Restore web discovery connectivity, capture one request that yields production candidates, compare production and popularity orders on that identical candidate set where existing judgments cover it, then stop. Keep the trace outside version control. If judgment coverage is insufficient, report coverage, leave ranking metrics suppressed, and stop.
"""


def main() -> None:
    report = build_report()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "evaluation_status.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"Report written to {OUTPUT}")


if __name__ == "__main__":
    main()
