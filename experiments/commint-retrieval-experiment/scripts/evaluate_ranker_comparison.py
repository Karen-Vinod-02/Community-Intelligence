"""Compare saved ranking baselines with the standalone experiment rankers.

Only existing study_accountability candidates and relevance labels are read.
The script does not call APIs or modify source artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

from evaluation.metrics import ndcg_at_k, reciprocal_rank
from evaluation.rank import rank_communities, rank_threads


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "results" / "ranker_comparison_20261004"
CASE_ID = "study_accountability"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _annotations(path: Path):
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = row.get("relevance", "").strip()
            if not raw:
                continue
            key = (row["item_type"].strip(), row["item_id"].strip().lower().removeprefix("r/"))
            if key in result:
                raise ValueError(f"Duplicate annotation: {key}")
            value = int(raw)
            if not 0 <= value <= 3:
                raise ValueError(f"Invalid relevance label {value} for {key}")
            result[key] = value
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_order(ids: list[str], item_type: str, annotations: dict):
    """Compute rank metrics only for a unique, fully judged list."""
    normalized = [str(value).strip().lower().removeprefix("r/") for value in ids]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"Duplicate {item_type} IDs in ranked output")
    labels = [annotations.get((item_type, value)) for value in normalized]
    coverage = sum(value is not None for value in labels)
    complete = bool(normalized) and coverage == len(normalized)
    judged_labels = [value for value in labels if value is not None]
    return {
        "candidate_count": len(normalized),
        "judged_count": coverage,
        "unjudged_count": len(normalized) - coverage,
        "judgment_coverage": coverage / len(normalized) if normalized else None,
        "judgments_complete": complete,
        "relevance_distribution": dict(sorted(Counter(judged_labels).items())),
        "nDCG@10": ndcg_at_k(judged_labels, 10) if complete else None,
        "MRR": reciprocal_rank(judged_labels) if complete else None,
        "ranked_ids": normalized,
    }


def build_comparison():
    case_dir = DATA / "normalized" / "latest" / CASE_ID
    annotations_path = DATA / "annotations" / "latest" / "annotations.csv"
    community_path = case_dir / "communities.json"
    thread_path = case_dir / "threads.json"
    ranker_path = ROOT / "evaluation" / "rank.py"
    metric_path = ROOT / "evaluation" / "metrics.py"
    case_config_path = ROOT / "cases" / "retrieval_cases.json"

    annotations = _annotations(annotations_path)
    community_data = _json(community_path)
    thread_data = _json(thread_path)
    thread_candidates = thread_data["selected"]
    if not isinstance(thread_candidates, list):
        raise ValueError("Saved selected thread candidates are not a list")

    baseline_thread_ids = [str(item.get("post_id") or item.get("id") or "") for item in thread_candidates]
    if any(not value for value in baseline_thread_ids):
        raise ValueError("A selected thread is missing its annotation ID")
    relevance_ranked_threads = rank_threads(thread_candidates)
    reranked_thread_ids = [str(item.get("post_id") or item.get("id") or "") for item in relevance_ranked_threads]

    thread_rank_details = []
    for rank, item in enumerate(relevance_ranked_threads, start=1):
        post_id = str(item.get("post_id") or item.get("id") or "")
        thread_rank_details.append({
            "rank": rank,
            "post_id": post_id,
            "subreddit": item.get("subreddit"),
            "_ranking_score": item["_ranking_score"],
            "_useful_comment_count": item["_useful_comment_count"],
            "relevance": annotations.get(("thread", post_id.lower())),
            "baseline_num_comments": item.get("num_comments", 0),
            "baseline_post_score": item.get("score", 0),
        })

    represented_subreddits = set(
        str(item.get("subreddit", "")).strip().lower().removeprefix("r/")
        for item in thread_candidates
    )
    baseline_communities = [
        item for item in community_data["ranked"]
        if item["subreddit"].strip().lower().removeprefix("r/") in represented_subreddits
    ]
    baseline_community_ids = [item["subreddit"] for item in baseline_communities]
    reranked_communities = rank_communities(thread_candidates)
    reranked_community_ids = [item["subreddit"] for item in reranked_communities]
    community_details = []
    for rank, item in enumerate(reranked_communities, start=1):
        name = item["subreddit"]
        baseline = next(row for row in baseline_communities if row["subreddit"].lower() == name.lower())
        community_details.append({
            "rank": rank,
            "subreddit": name,
            "thread_count": item["thread_count"],
            "evidence_comment_count_length_ge_20": item["evidence_comment_count"],
            "total_comments_reported_on_posts": item["total_comments"],
            "baseline_mcp_rank": baseline["combined_rank"],
            "baseline_mcp_score": baseline["best_source_score"],
            "relevance": annotations.get(("community", name.lower())),
        })

    comments = [comment for item in thread_candidates for comment in item.get("comments", [])]
    body_comments = [comment for comment in comments if str(comment.get("body") or "").strip()]
    long_comments = [comment for comment in body_comments if len(str(comment.get("body") or "").strip()) >= 20]
    thread_evidence = {
        "selected_thread_candidates": len(thread_candidates),
        "threads_with_nonempty_comment_lists": sum(bool(item.get("comments")) for item in thread_candidates),
        "saved_comment_records": len(comments),
        "comments_with_nonempty_body": len(body_comments),
        "comments_with_body_length_at_least_20": len(long_comments),
        "threads_with_title_and_text": sum(bool(item.get("title", "").strip()) and bool(item.get("text", "").strip()) for item in thread_candidates),
        "threads_with_urls": sum(bool(item.get("url")) for item in thread_candidates),
        "retrieval_source_counts": dict(Counter(item.get("source", "unknown") for item in thread_candidates)),
        "discovery_source_counts": dict(Counter(source for item in thread_candidates for source in item.get("discovery_sources", []))),
        "per_call_retrieval_success_logs_available": False,
        "evidence_usefulness_judgments_available": False,
    }

    source_files = {
        "annotations": annotations_path,
        "normalized_communities": community_path,
        "selected_thread_pool_and_comments": thread_path,
        "ranker_code": ranker_path,
        "metric_code": metric_path,
        "case_configuration": case_config_path,
    }
    production_model_installed = importlib.util.find_spec("sentence_transformers") is not None

    return {
        "generated_on": "2026-10-04",
        "case_id": CASE_ID,
        "sources": {key: {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)} for key, path in source_files.items()},
        "ranker_scope": {
            "thread_ranker": "evaluation.rank.rank_threads; standalone experimental function, not called by run_experiment.py or the production API.",
            "community_ranker": "evaluation.rank.rank_communities; standalone experimental aggregation, not called by the production API.",
            "production_community_ranker": "backend.app.services.ranking.CommunityRanker is called by the production /api/analyze path after CommunityFilter. It was not replayed: the saved exact product problem description and full per-community retrieved post inputs are unavailable, and sentence_transformers is not installed in this environment.",
            "sentence_transformers_available": production_model_installed,
            "thread_ranker_characterization": "Despite being described as relevance ranking, thread_score uses post score, reported comment count, and positive scores from comments whose body length is at least 20 characters. It has no query/problem text or semantic relevance input.",
            "community_ranker_characterization": "rank_communities orders by count of comments with body length at least 20, then thread count, then reported total comments. It does not inspect comment meaning or relevance labels.",
        },
        "thread_comparison": {
            "candidate_set_source": "normalized/latest/study_accountability/threads.json:selected",
            "candidate_pool_scope": "The saved selected top-15 pool only; the pre-truncation retrieved pool is absent.",
            "candidate_coverage": {
                "baseline_items": len(baseline_thread_ids),
                "reranker_output_items": len(reranked_thread_ids),
                "reranker_retained_candidate_fraction": len(reranked_thread_ids) / len(baseline_thread_ids) if baseline_thread_ids else None,
            },
            "baseline_tie_rule": "Existing selected list uses descending (num_comments, score); stable order resolves remaining ties.",
            "reranker_tie_rule": "Descending _ranking_score; Python stable sort retains incoming baseline order for ties.",
            "baseline": evaluate_order(baseline_thread_ids, "thread", annotations),
            "comment_engagement_reranker": evaluate_order(reranked_thread_ids, "thread", annotations),
            "reranker_ranked_scores": thread_rank_details,
            "ranker_inputs": thread_candidates,
            "comparison_interpretation": "Same 15 candidates and same 15 relevance labels. This compares the saved popularity order with a comment-engagement heuristic, not with a semantic relevance ranker.",
        },
        "community_comparison": {
            "candidate_set": sorted(represented_subreddits),
            "candidate_set_scope": "Only communities represented among the saved 15 selected threads (3 communities); the other 2 of 5 selected communities have no thread in this saved selected list, and 9 of 14 ranked communities have no saved thread inputs.",
            "candidate_coverage": {
                "baseline_items": len(baseline_community_ids),
                "reranker_output_items": len(reranked_community_ids),
                "reranker_retained_candidate_fraction": len(reranked_community_ids) / len(baseline_community_ids) if baseline_community_ids else None,
            },
            "baseline_tie_rule": "Normalized MCP combined_rank, restricted to the same three represented communities.",
            "reranker_tie_rule": "Descending (evidence_comment_count, thread_count, total_comments); stable first-seen order resolves exact ties.",
            "baseline": evaluate_order(baseline_community_ids, "community", annotations),
            "comment_aggregation_ranker": evaluate_order(reranked_community_ids, "community", annotations),
            "ranked_scores": community_details,
            "comparison_interpretation": "Same three represented communities and same three existing labels. This is a small selected-subset comparison of engagement/evidence-volume ordering, not the production semantic CommunityRanker.",
        },
        "evidence_availability": thread_evidence,
        "production_path": [
            "POST /api/analyze calls CommunityService.analyze.",
            "By default discovery uses WebSearchDiscoverySource (DDGS plus embedding relevance threshold) and Reddit retrieval uses ArcticShiftSource. CommunityService extracts queries, deduplicates candidates, caps discovery at 15 communities, requests twice posts_per_community, filters unusable/old/wrong-subreddit posts, then applies CommunityFilter semantic reranking and CommunityRanker evidence/relevance scoring.",
            "The API returns up to five results whose signals.problem_relevant_posts is positive and their top posts. CommunityRanker can emit a potential tier, but the route constructs CommunityResult without forwarding status or signals and does not set AnalyzeResponse.fallback_used; response defaults can therefore label potential results as relevant and report no fallback. Production API code does not fetch comments.",
            "The experimental run_experiment.py uses a separate confidence/query-coverage MCP community baseline, retrieves Arctic Shift threads, sorts by (num_comments, score), selects 15, then fetches comments. It does not call evaluation.rank.rank_threads or rank_communities.",
        ],
        "limitations": [
            "No full pre-truncation thread candidate pool or per-call retrieval logs are saved; retrieval success rate and recall cannot be computed from the normalized artifacts.",
            "Comments exist in the selected thread artifact, but there are no comment-level usefulness labels or evidence-span annotations. Text length/counts establish availability only, not evidence quality.",
            "Thread relevance labels are all positive (11 label-3, 4 label-2), so MRR is necessarily 1.0 for any complete ordering of these candidates; nDCG is the informative ordering measure here.",
            "Community subset has only three candidates, all relevance-positive (two 3s and one 2), and is selected through the existing top-15 thread output.",
            "The API response currently drops CommunityRanker status/signals and does not set fallback_used, so a potential-tier result can be presented with the response model's default relevant status and fallback_used=false.",
            "Production backend ranker cannot be fairly evaluated without the original product description and full retrieved post pool for the same 14 judged communities; its embedding dependency is also unavailable here.",
        ],
    }


def render_report(data: dict) -> str:
    threads = data["thread_comparison"]
    communities = data["community_comparison"]
    evidence = data["evidence_availability"]
    thread_rows = "\n".join(
        f"| {row['rank']} | `{row['post_id']}` | {row['_ranking_score']:.6f} | {row['relevance']} |"
        for row in threads["reranker_ranked_scores"]
    )
    community_rows = "\n".join(
        f"| {row['rank']} | r/{row['subreddit']} | {row['baseline_mcp_rank']} | {row['evidence_comment_count_length_ge_20']} | {row['thread_count']} | {row['relevance']} |"
        for row in communities["ranked_scores"]
    )
    metric_table = "\n".join([
        "| Candidate set / order | Candidates | Judged | Label distribution | nDCG@10 | MRR |",
        "|---|---:|---:|---|---:|---:|",
        f"| Threads / popularity baseline | {threads['baseline']['candidate_count']} | {threads['baseline']['judged_count']} | {threads['baseline']['relevance_distribution']} | {threads['baseline']['nDCG@10']:.4f} | {threads['baseline']['MRR']:.4f} |",
        f"| Threads / comment-engagement reranker | {threads['comment_engagement_reranker']['candidate_count']} | {threads['comment_engagement_reranker']['judged_count']} | {threads['comment_engagement_reranker']['relevance_distribution']} | {threads['comment_engagement_reranker']['nDCG@10']:.4f} | {threads['comment_engagement_reranker']['MRR']:.4f} |",
        f"| Communities / MCP baseline, restricted to same 3 | {communities['baseline']['candidate_count']} | {communities['baseline']['judged_count']} | {communities['baseline']['relevance_distribution']} | {communities['baseline']['nDCG@10']:.4f} | {communities['baseline']['MRR']:.4f} |",
        f"| Communities / comment-aggregation ranker | {communities['comment_aggregation_ranker']['candidate_count']} | {communities['comment_aggregation_ranker']['judged_count']} | {communities['comment_aggregation_ranker']['relevance_distribution']} | {communities['comment_aggregation_ranker']['nDCG@10']:.4f} | {communities['comment_aggregation_ranker']['MRR']:.4f} |",
    ])
    return f"""# Saved Ranker Comparison — {data['generated_on']}

## Scope

This evaluation uses only existing study_accountability artifacts and labels. The popularity and experimental rankers below receive the same candidate IDs and judgments within each comparison. Source hashes and full thread ranker inputs are in `comparison.json`.

## Production path and ranker usage

The production API is `POST /api/analyze`. By default it discovers via web search, retrieves from Arctic Shift, applies recent/usable/subreddit filters, semantically reranks candidates with `CommunityFilter`, then invokes backend `CommunityRanker`. It returns up to five results with positive qualifying-post counts and top posts. The production API path does not fetch comments. The route omits ranker `status` and `signals` and does not set `fallback_used`, so potential-tier results can inherit the response model's default `relevant` status and `fallback_used=false`.

The experiment pipeline is separate: it ranks MCP communities using confidence/query coverage, retrieves Arctic Shift posts, selects 15 by descending `(num_comments, score)`, and fetches comments afterward. It does not invoke `evaluation.rank.rank_threads` or `rank_communities`.

The standalone `rank_threads` function is not semantic relevance ranking: it combines post score, reported comment count, and upvoted comments with bodies at least 20 characters. `rank_communities` sorts by long-comment count, thread count, and reported comment totals. Neither reads problem text or comment meaning.

## Same-pool ranking comparison

{metric_table}

### Thread reranker order

The reranker retained all {threads['candidate_coverage']['reranker_output_items']}/{threads['candidate_coverage']['baseline_items']} candidates. It uses descending `_ranking_score`; stable sorting preserves baseline order on ties.

| Rank | Post ID | Reranker score | Relevance |
|---:|---|---:|---:|
{thread_rows}

Thread nDCG fell from {threads['baseline']['nDCG@10']:.4f} to {threads['comment_engagement_reranker']['nDCG@10']:.4f}; MRR stayed 1.0000. This does not show an improvement from the experimental comment-engagement heuristic. Since all 15 labels are positive (11×3, 4×2), MRR is 1.0 for every complete ordering; nDCG is the useful comparison metric.

### Community aggregation comparison

This is restricted to the three communities represented in the selected 15 threads. The baseline is the saved MCP ranking restricted to those same IDs. The ranker changed the order, but nDCG remained {communities['baseline']['nDCG@10']:.4f}; MRR is uninformative because all three labels are positive.

| Experimental rank | Community | Saved MCP rank | Long comments counted | Threads | Relevance |
|---:|---|---:|---:|---:|---:|
{community_rows}

Two of the five selected MCP communities have no thread in the saved selected list, and nine of the 14 ranked communities have no saved thread inputs. This cannot evaluate community ranking over the full judged community set.

## Evidence and retrieval measurements

- Saved pool: {evidence['selected_thread_candidates']} selected thread candidates; the full pre-truncation pool is missing.
- Non-empty comment lists: {evidence['threads_with_nonempty_comment_lists']}/{evidence['selected_thread_candidates']} threads.
- Saved comment records: {evidence['saved_comment_records']}; non-empty bodies: {evidence['comments_with_nonempty_body']}; bodies at least 20 characters: {evidence['comments_with_body_length_at_least_20']}.
- Thread title and text: {evidence['threads_with_title_and_text']}/{evidence['selected_thread_candidates']}; URLs: {evidence['threads_with_urls']}/{evidence['selected_thread_candidates']}.
- Source provenance: retrieval {evidence['retrieval_source_counts']}; discovery {evidence['discovery_source_counts']}.
- Per-call retrieval success logs: unavailable. Evidence usefulness labels/spans: unavailable.

Comment counts and body length measure availability only. They do not establish that comments contain useful supporting evidence. In the experiment pipeline comments are fetched after the popularity-based selection; only the standalone offline `rank_threads` function uses saved comments to rerank.

## Production-ranker evaluation boundary

The backend `CommunityRanker` and preceding `CommunityFilter` were not executed on these artifacts. The exact user problem description is absent; the case file has retrieval queries, which are not equivalent inputs. The normalized data has MCP candidates for 14 communities but thread content for only three, with only the already selected 15 posts. A fair same-pool comparison against 14 community judgments therefore lacks both ranker input and candidate coverage. `sentence_transformers` is unavailable in this environment. No model output is simulated.

## Reproduction and remaining work

Run from this directory: `python evaluate_ranker_comparison.py`. It writes JSON and Markdown to `results/ranker_comparison_20261004/` and performs no network calls.

Next system evaluation should capture the exact user problem description, the common pre-ranking community/post candidate pools, production filter/ranker outputs and scores, plus per-call retrieval logs. Then compare the production ordering with the same popularity baseline over identical candidates and existing judgments. Separately, evidence usefulness requires evidence-level judgments; comment volume is not a substitute.
"""


def main():
    result = build_comparison()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT / "report.md").write_text(render_report(result), encoding="utf-8")
    print(f"Ranker comparison written to {OUTPUT}")


if __name__ == "__main__":
    main()
