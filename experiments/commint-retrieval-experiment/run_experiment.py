import argparse
import json
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from sources.reddit_mcp import RedditMCPSource
from sources.agent_reach import AgentReachSource
from sources.arctic_shift import ArcticShiftSource

from evaluation.normalize import (
    group_communities_by_source,
    deduplicate_communities,
    rank_communities,
    select_top_communities,
    deduplicate_threads,
)

from config import ARCTIC_COMMENT_LIMIT


ROOT = Path(__file__).resolve().parent

RAW_DIR = ROOT / "data" / "raw" / "latest"
NORMALIZED_DIR = ROOT / "data" / "normalized" / "latest"


def load_cases():
    path = ROOT / "cases" / "retrieval_cases.json"

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def save_json(path, data):
    """
    Atomically write JSON.

    The data is first written to a temporary file.
    Only after serialization succeeds is the temporary file
    moved into place.

    This prevents a failed run from leaving a 0-byte JSON file.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    serialized = json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    temporary_path.write_text(
        serialized,
        encoding="utf-8",
    )

    os.replace(
        temporary_path,
        path,
    )


def save_source_results(
    source_dir,
    case_id,
    candidates,
):
    save_json(
        source_dir / f"{case_id}.json",
        [
            asdict(candidate)
            for candidate in candidates
        ],
    )


def save_source_error(
    source_dir,
    case_id,
    exc,
):
    save_json(
        source_dir / f"{case_id}_error.json",
        {
            "error": str(exc),
            "exception_type": type(exc).__name__,
        },
    )


# ==========================================================
# TARGETED THREAD-RETRIEVAL RETRY
#
# Use this when community discovery already succeeded and
# most communities returned threads fine, but one or two
# communities failed thread retrieval (e.g. Arctic Shift
# 422s) and you don't want to re-run:
#
#   - MCP / Agent-Reach discovery
#   - community ranking
#   - community annotations
#   - thread retrieval for communities that already worked
#
# It re-runs thread + comment retrieval ONLY for the named
# subreddits, then merges the results into the existing
# threads.json rather than overwriting it wholesale.
# ==========================================================

def retry_thread_retrieval(case, target_subreddits):
    case_id = case["case_id"]

    target_subreddits = [
        s.strip().lower()
        for s in target_subreddits
        if s.strip()
    ]

    community_path = (
        NORMALIZED_DIR / case_id / "communities.json"
    )

    threads_path = (
        NORMALIZED_DIR / case_id / "threads.json"
    )

    if not community_path.exists():
        raise RuntimeError(
            f"{case_id}: communities.json not found. "
            f"Run full discovery for this case first "
            f"(python run_experiment.py, no --communities flag)."
        )

    communities_data = json.loads(
        community_path.read_text(encoding="utf-8")
    )

    known_subreddits = {
        c["subreddit"]
        for c in communities_data.get("deduplicated", [])
    }

    unknown = [
        s for s in target_subreddits
        if s not in known_subreddits
    ]

    if unknown:
        raise RuntimeError(
            f"{case_id}: these subreddits are not in the "
            f"existing community-discovery pool for this case: "
            f"{unknown}. Known subreddits: "
            f"{sorted(known_subreddits)}"
        )

    # --------------------------------------------------
    # Load whatever threads already exist, so we merge
    # rather than clobber.
    # --------------------------------------------------

    existing_selected = []
    thread_query_count = 0

    if threads_path.exists() and threads_path.stat().st_size > 0:
        existing_data = json.loads(
            threads_path.read_text(encoding="utf-8")
        )

        if isinstance(existing_data, dict):
            existing_selected = existing_data.get(
                "selected", []
            )
            thread_query_count = existing_data.get(
                "thread_query_count", 0
            )

        elif isinstance(existing_data, list):
            # Older bare-list format.
            existing_selected = existing_data

    target_set = set(target_subreddits)

    # Drop any existing entries for the subreddits being
    # retried, in case they hold stale/partial data (e.g.
    # 0 threads from a prior 422).
    kept_existing = [
        t for t in existing_selected
        if str(t.get("subreddit", "")).lower() not in target_set
    ]

    dropped_count = len(existing_selected) - len(kept_existing)

    print()
    print(
        f"[retry] Case {case_id}: retrying thread retrieval "
        f"for {target_subreddits}"
    )

    print(
        f"[retry] Existing threads kept from untouched "
        f"communities: {len(kept_existing)} "
        f"(dropped {dropped_count} stale entries for "
        f"targeted communities)"
    )

    # --------------------------------------------------
    # Re-fetch threads for the target subreddits only.
    # --------------------------------------------------

    arctic = ArcticShiftSource()

    new_threads = []

    for index, subreddit in enumerate(target_subreddits):

        if index > 0:
            time.sleep(arctic.REQUEST_DELAY_SECONDS)

        print(f"[retry] r/{subreddit}: retrieving threads...")

        try:
            threads = arctic.retrieve_threads(case, subreddit)

        except Exception as exc:
            print(f"  THREAD RETRIEVAL FAILED: {exc}")
            continue

        print(f"  retrieved {len(threads)} thread candidates")

        for thread in threads:
            thread.discovery_sources = ["reddit_mcp"]

        new_threads.extend(threads)

    if not new_threads:
        print(
            "[retry] No threads retrieved for any target "
            "community. threads.json left unchanged."
        )
        return

    # --------------------------------------------------
    # Comments, only for the newly retrieved threads.
    # --------------------------------------------------

    comment_limit = int(
        case.get(
            "comments_per_thread",
            ARCTIC_COMMENT_LIMIT,
        )
    )

    print()
    print("[retry] Fetching comments for new threads...")

    for index, thread in enumerate(new_threads, start=1):
        print(
            f"  {index}/{len(new_threads)} "
            f"{thread.post_id}"
        )

        try:
            comments = arctic.fetch_comments(thread.post_id)
            thread.comments = comments[:comment_limit]

            print(f"    retrieved {len(thread.comments)} comments")

        except Exception as exc:
            print(f"    COMMENT RETRIEVAL FAILED: {exc}")
            thread.comments = []

    # --------------------------------------------------
    # Merge, dedupe, re-clip to top_n.
    #
    # NOTE: this re-sorts the combined pool using the same
    # baseline ordering as the full pipeline (num_comments,
    # then score), but it does NOT re-run the original
    # cross-community selection from scratch, since we did
    # not re-fetch candidates for the untouched communities.
    # This is a close approximation, not a byte-identical
    # reproduction of a full re-run.
    # --------------------------------------------------

    new_thread_dicts = [
        asdict(thread) for thread in new_threads
    ]

    combined = kept_existing + new_thread_dicts

    seen_ids = set()
    deduped = []

    for thread in combined:
        post_id = thread.get("post_id")

        if post_id in seen_ids:
            continue

        if post_id:
            seen_ids.add(post_id)

        deduped.append(thread)

    deduped.sort(
        key=lambda t: (
            t.get("num_comments", 0),
            t.get("score", 0),
        ),
        reverse=True,
    )

    top_n = int(case.get("top_threads", 15))

    selected = deduped[:top_n]

    save_json(
        threads_path,
        {
            "case_id": case_id,
            "thread_query_count": thread_query_count or 1,
            "selected": selected,
        },
    )

    print()
    print(
        f"[retry] Updated threads.json: {len(selected)} threads "
        f"({len(new_thread_dicts)} newly retrieved, merged with "
        f"{len(kept_existing)} kept from untouched communities)."
    )

    print(
        "[retry] Now re-run build_pool.py and merge_annotations.py "
        "to fold this into your annotation pool."
    )


# ==========================================================
# FULL PIPELINE (unchanged from before)
# ==========================================================

def run():
    cases = load_cases()

    mcp = RedditMCPSource()
    agent_reach = AgentReachSource()
    arctic = ArcticShiftSource()

    for case in cases:

        case_id = case["case_id"]

        print_separator()
        print(f"CASE: {case_id}")
        print_separator()

        # ==================================================
        # COMMUNITY DISCOVERY
        # ==================================================

        mcp_results = []
        agent_results = []

        # --------------------------------------------------
        # Reddit MCP
        # --------------------------------------------------

        print()
        print("[MCP] discovering communities...")

        try:
            mcp_results = mcp.discover(case)

            print(
                f"[MCP] returned "
                f"{len(mcp_results)} results."
            )

            save_source_results(
                RAW_DIR / "reddit_mcp",
                case_id,
                mcp_results,
            )

        except Exception as exc:
            print()
            print("[MCP] FAILED")
            print(f"Error: {exc}")
            traceback.print_exc()

            save_source_error(
                RAW_DIR / "reddit_mcp",
                case_id,
                exc,
            )

        # --------------------------------------------------
        # Agent Reach
        # --------------------------------------------------

        print()
        print(
            "[Agent-Reach] supplementary Reddit retrieval..."
        )

        try:
            agent_results = agent_reach.discover(case)

            print(
                f"[Agent-Reach] returned "
                f"{len(agent_results)} results."
            )

            save_source_results(
                RAW_DIR / "agent_reach",
                case_id,
                agent_results,
            )

        except Exception as exc:
            print()
            print(
                "[Agent-Reach] "
                "UNAVAILABLE / FAILED"
            )

            print(f"Error: {exc}")
            traceback.print_exc()

            save_source_error(
                RAW_DIR / "agent_reach",
                case_id,
                exc,
            )

        # ==================================================
        # COMMUNITY DISCOVERY CHECK
        # ==================================================

        if not mcp_results:

            print_separator()

            print(
                "ERROR: Reddit MCP did not return "
                "community candidates."
            )

            print(
                "Arctic Shift will NOT be called."
            )

            print_separator()

            continue

        # ==================================================
        # SOURCE-SPECIFIC RANKING
        # ==================================================

        print()
        print("SOURCE RANKINGS")

        by_source = group_communities_by_source(
            mcp_results
        )

        for source, items in by_source.items():

            print()
            print(source)

            for item in items:
                print(
                    f"  "
                    f"{item.source_rank}. "
                    f"r/{item.subreddit} "
                    f"(score="
                    f"{item.source_score:.3f})"
                )

        # Agent Reach is reported separately.
        # It is NOT merged into the community-discovery pool.
        if agent_results:

            agent_by_source = group_communities_by_source(
                agent_results
            )

            for source, items in agent_by_source.items():

                print()
                print(
                    f"{source} "
                    "(supplementary retrieval)"
                )

                for item in items:
                    print(
                        f"  "
                        f"{item.source_rank}. "
                        f"r/{item.subreddit} "
                        f"(score="
                        f"{item.source_score:.3f})"
                    )

        # ==================================================
        # COMMUNITY POOL
        # ==================================================

        # IMPORTANT:
        # Only MCP candidates enter the community-ranking
        # and Arctic Shift pipeline.
        #
        # Agent Reach remains available as a separate source
        # for source comparison, but its noisy post-search
        # results do not become community candidates.
        community_pool = deduplicate_communities(
            mcp_results
        )

        if not community_pool:

            print()
            print(
                "ERROR: MCP community deduplication "
                "produced an empty pool."
            )

            continue

        print()
        print("DEDUPLICATED COMMUNITY POOL")

        for community in community_pool:

            print(
                f"  "
                f"r/{community['subreddit']} "
                f"<- "
                f"{community['sources']}"
            )

        # ==================================================
        # COMBINED COMMUNITY RANKING
        # ==================================================

        ranked_communities = rank_communities(
            community_pool
        )

        top_community_limit = int(
            case.get(
                "top_communities",
                5,
            )
        )

        selected_communities = select_top_communities(
            ranked_communities,
            top_community_limit,
        )

        print()
        print("COMBINED COMMUNITY RANKING")

        for community in ranked_communities:

            print(
                f"  "
                f"{community['combined_rank']}. "
                f"r/{community['subreddit']} "
                f"| confidence="
                f"{community['best_source_score']:.3f} "
                f"| query_coverage="
                f"{community['query_coverage']} "
                f"| best_source_rank="
                f"{community['best_source_rank']}"
            )

        # ==================================================
        # SELECTED COMMUNITIES
        # ==================================================

        print()
        print("SELECTED COMMUNITIES")

        for index, community in enumerate(
            selected_communities,
            start=1,
        ):

            print(
                f"  "
                f"{index}. "
                f"r/{community['subreddit']} "
                f"(combined_rank="
                f"{community['combined_rank']})"
            )

        # ==================================================
        # SAVE COMMUNITY DATA
        # ==================================================

        community_output = {
            "case_id": case_id,

            # MCP is the community discovery source.
            "community_discovery_source": "reddit_mcp",

            # MCP candidates are the actual community pool.
            "all_candidates": [
                asdict(candidate)
                for candidate in mcp_results
            ],

            "deduplicated": community_pool,

            "ranked": ranked_communities,

            "selected": selected_communities,

            # Keep Agent Reach separately so it can be
            # analyzed without polluting the community pool.
            "supplementary_sources": {
                "agent_reach": [
                    asdict(candidate)
                    for candidate in agent_results
                ]
            },
        }

        save_json(
            NORMALIZED_DIR
            / case_id
            / "communities.json",
            community_output,
        )

        print()
        print(
            f"Selected top "
            f"{len(selected_communities)} "
            "communities for thread retrieval."
        )

        # ==================================================
        # THREAD RETRIEVAL
        # ==================================================

        all_threads = []

        thread_queries = case.get(
            "thread_queries"
        )

        if not thread_queries:

            discovery_queries = case.get(
                "queries",
                [],
            )

            thread_queries = (
                discovery_queries[:1]
                if discovery_queries
                else []
            )

        print()
        print("[Arctic Shift] retrieving threads...")

        print("Thread queries:")

        for query in thread_queries:
            print(f"  - {query}")

        print(
            "This pilot deliberately uses "
            f"{len(thread_queries)} "
            "thread query per community."
        )

        for community_index, community in enumerate(
            selected_communities,
            start=1,
        ):

            subreddit = community["subreddit"]

            print()
            print(
                f"[{community_index}/"
                f"{len(selected_communities)}] "
                f"r/{subreddit}"
            )

            try:

                threads = arctic.retrieve_threads(
                    case,
                    subreddit,
                )

            except Exception as exc:

                print(
                    "    THREAD RETRIEVAL FAILED: "
                    f"{exc}"
                )

                continue

            print(
                f"    retrieved "
                f"{len(threads)} "
                "thread candidates"
            )

            for thread in threads:

                thread.discovery_sources = list(
                    community["sources"]
                )

            all_threads.extend(threads)

        # ==================================================
        # THREAD DEDUPLICATION
        # ==================================================

        unique_threads = deduplicate_threads(
            all_threads
        )

        print()
        print(
            f"Unique thread candidates: "
            f"{len(unique_threads)}"
        )

        # ==================================================
        # SELECT TOP THREADS
        # ==================================================

        top_n = int(
            case.get(
                "top_threads",
                15,
            )
        )

        selected_threads = unique_threads[:top_n]

        print()
        print(
            f"Selected "
            f"{len(selected_threads)} "
            "threads for comment retrieval."
        )

        # ==================================================
        # COMMENT RETRIEVAL
        # ==================================================

        comment_limit = int(
            case.get(
                "comments_per_thread",
                ARCTIC_COMMENT_LIMIT,
            )
        )

        comment_failures = 0
        threads_with_comments = 0
        total_comments = 0

        print()
        print("FETCHING COMMENTS")

        for index, thread in enumerate(
            selected_threads,
            start=1,
        ):

            print(
                f"  {index}/"
                f"{len(selected_threads)} "
                f"{thread.post_id} "
                f"({thread.num_comments} "
                "comments reported by Reddit)"
            )

            try:

                comments = arctic.fetch_comments(
                    thread.post_id
                )

                thread.comments = comments[
                    :comment_limit
                ]

                retrieved_count = len(
                    thread.comments
                )

                if retrieved_count > 0:

                    threads_with_comments += 1
                    total_comments += retrieved_count

                print(
                    f"    retrieved "
                    f"{retrieved_count} "
                    "comments"
                )

            except Exception as exc:

                comment_failures += 1

                print(
                    "    COMMENT RETRIEVAL FAILED: "
                    f"{exc}"
                )

                thread.comments = []

        # ==================================================
        # SAVE THREAD DATA
        # ==================================================

        save_json(
            NORMALIZED_DIR
            / case_id
            / "threads.json",
            {
                "case_id": case_id,
                "thread_query_count": len(
                    thread_queries
                ),
                "selected": [
                    asdict(thread)
                    for thread in selected_threads
                ],
            },
        )

        # ==================================================
        # FINAL SUMMARY
        # ==================================================

        print()
        print("-" * 70)

        print(
            f"CASE COMPLETE: {case_id}"
        )

        print(
            f"MCP discovery candidates: "
            f"{len(mcp_results)}"
        )

        print(
            f"Agent-Reach supplementary candidates: "
            f"{len(agent_results)}"
        )

        print(
            f"Unique MCP communities: "
            f"{len(community_pool)}"
        )

        print(
            f"Communities sent to Arctic Shift: "
            f"{len(selected_communities)}"
        )

        print(
            f"Thread queries per community: "
            f"{len(thread_queries)}"
        )

        print(
            f"Unique thread candidates: "
            f"{len(unique_threads)}"
        )

        print(
            f"Selected threads: "
            f"{len(selected_threads)}"
        )

        print(
            f"Threads with retrieved comments: "
            f"{threads_with_comments}"
        )

        print(
            f"Total comments retrieved: "
            f"{total_comments}"
        )

        print(
            f"Comment retrieval failures: "
            f"{comment_failures}"
        )

        print("-" * 70)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the full retrieval pipeline, or, with "
            "--communities, retry ONLY thread+comment "
            "retrieval for specific subreddits in one case "
            "(community discovery/ranking is left untouched)."
        )
    )

    parser.add_argument(
        "--communities",
        nargs="+",
        default=None,
        metavar="SUBREDDIT",
        help=(
            "One or more subreddit names to retry thread "
            "retrieval for (e.g. --communities getstudying "
            "motivation). Requires --case. Does NOT run "
            "community discovery/ranking; the subreddits "
            "must already exist in that case's "
            "communities.json."
        ),
    )

    parser.add_argument(
        "--case",
        default=None,
        metavar="CASE_ID",
        help=(
            "case_id to target (required when using "
            "--communities). Must match a case_id in "
            "cases/retrieval_cases.json."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.communities:
        if not args.case:
            raise SystemExit(
                "--communities requires --case CASE_ID"
            )

        cases = load_cases()

        matching = [
            c for c in cases
            if c["case_id"] == args.case
        ]

        if not matching:
            available = [c["case_id"] for c in cases]

            raise SystemExit(
                f"No case with case_id={args.case!r}. "
                f"Available: {available}"
            )

        retry_thread_retrieval(
            matching[0],
            args.communities,
        )

    else:
        run()