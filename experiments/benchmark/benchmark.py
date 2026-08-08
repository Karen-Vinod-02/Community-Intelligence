"""
Run the full PullPush, Reddit API(PRAW), and  Arctic Shift benchmark 

Usage:
    pip install -r ../requirements.txt  
    python benchmark.py

What this measures, mapped to the four open questions:
  1. PullPush: subreddit discovery via keyword search
     -> discover() calls, ranked by subreddit match count, success/failure rate
  2. Arctic Shift: post/comment retrieval
     -> fetch_posts() on the top subreddits PullPush surfaced, then
        fetch_comments() from the first returned post, to confirm the full
        discover -> retrieve -> comments pipeline actually chains together
  3. Date filtering (last ~90 days)
     -> both clients apply the date window; log no. of results returned
  4. Latency
     -> per-call latency + aggregate p50/p95/avg, per provider

Everything is saved to results/*.json
"""
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import TEST_QUERIES, KNOWN_SUBREDDITS, DAYS_WINDOW, RESULTS_PER_QUERY, RESULTS_DIR
from metrics import summarize
from clients import pullpush, arcticshift, reddit_praw


def ensure_results_dir():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULTS_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def save_json(results_dir, filename, payload):
    with open(os.path.join(results_dir, filename), "w") as f:
        json.dump(payload, f, indent=2, default=str)


def run_pullpush_discovery(results_dir):
    print("\n=== PullPush: discovery benchmark ===")
    print("  (NOTE: PullPush's own maintenance notice says it's offline until")
    print("   end of September 2026 for hardware upgrades — expect all-502s.")
    print("   Kept in this benchmark only so the comparison data stays honest.)")
    all_results = []
    for query in TEST_QUERIES:
        print(f"  querying: {query!r} ...", end=" ", flush=True)
        result = pullpush.discover(query, days_window=DAYS_WINDOW, size=RESULTS_PER_QUERY)
        all_results.append(result)
        status = "OK" if result.ok else f"FAILED ({result.error})"
        print(f"{status} | {result.latency_seconds}s | {result.num_results} results")
        time.sleep(2.5)  # be polite to the free tier (~30 req/min ceiling)

    save_json(results_dir, "pullpush_discovery_raw.json", [r.to_dict() for r in all_results])
    return all_results


def run_reddit_official_discovery(results_dir):
    print("\n=== Reddit Official API (PRAW): discovery benchmark ===")
    all_results = []
    for i, query in enumerate(TEST_QUERIES):
        print(f"  querying: {query!r} ...", end=" ", flush=True)
        result = reddit_praw.discover(query, days_window=DAYS_WINDOW, size=RESULTS_PER_QUERY)

        # discover() catches its own exceptions and reports them inside the
        # CallResult rather than raising — so check the first result for a
        # setup problem (missing library or missing credentials) and bail
        # out of the whole provider early, instead of repeating the same
        # failure for every remaining query.
        setup_problem_markers = ("is not installed", "Missing REDDIT")
        if i == 0 and not result.ok and result.error and any(m in result.error for m in setup_problem_markers):
            print(f"SKIPPED — {result.error}")
            return []

        all_results.append(result)
        status = "OK" if result.ok else f"FAILED ({result.error})"
        print(f"{status} | {result.latency_seconds}s | {result.num_results} results")
        time.sleep(1.0)  # official API is more generous (~100 req/min) but still be polite

    if all_results:
        save_json(results_dir, "reddit_official_discovery_raw.json", [r.to_dict() for r in all_results])
    return all_results


def run_arcticshift_retrieval(results_dir, candidate_subreddits):
    print("\n=== Arctic Shift: retrieval benchmark ===")
    post_results = []
    comment_results = []

    subs_to_test = candidate_subreddits or KNOWN_SUBREDDITS
    for sub in subs_to_test:
        print(f"  fetching posts from r/{sub} ...", end=" ", flush=True)
        post_result = arcticshift.fetch_posts(sub, days_window=DAYS_WINDOW, limit=RESULTS_PER_QUERY)
        post_results.append(post_result)
        status = "OK" if post_result.ok else f"FAILED ({post_result.error})"
        print(f"{status} | {post_result.latency_seconds}s | {post_result.num_results} posts")

        # If we got at least one post, test the comments endpoint too
        if post_result.ok and post_result.raw_sample:
            post_id = post_result.raw_sample[0].get("id")
            if post_id:
                link_id = post_id if post_id.startswith("t3_") else f"t3_{post_id}"
                print(f"    fetching comments for {link_id} ...", end=" ", flush=True)
                comment_result = arcticshift.fetch_comments(link_id)
                comment_results.append(comment_result)
                c_status = "OK" if comment_result.ok else f"FAILED ({comment_result.error})"
                print(f"{c_status} | {comment_result.latency_seconds}s | {comment_result.num_results} comments")

        time.sleep(0.5)

    save_json(results_dir, "arcticshift_posts_raw.json", [r.to_dict() for r in post_results])
    save_json(results_dir, "arcticshift_comments_raw.json", [r.to_dict() for r in comment_results])
    return post_results, comment_results


def top_candidate_subreddits(discovery_results, top_n=5):
    """Aggregate subreddit match counts across ALL discovery queries."""
    combined: dict[str, int] = {}
    for r in discovery_results:
        for sub, count in (r.subreddits_seen or []):
            combined[sub] = combined.get(sub, 0) + count
    ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
    return [sub for sub, _ in ranked[:top_n]]


def main():
    results_dir = ensure_results_dir()

    pullpush_results = run_pullpush_discovery(results_dir)
    reddit_official_results = run_reddit_official_discovery(results_dir)

    # Prefer whichever discovery source actually produced candidates —
    # right now that should be reddit_official, given PullPush's outage.
    discovery_results = reddit_official_results if reddit_official_results else pullpush_results
    discovery_source_used = "reddit_official" if reddit_official_results else "pullpush"

    candidates = top_candidate_subreddits(discovery_results)
    print(f"\nTop candidate subreddits from {discovery_source_used} discovery: "
          f"{candidates or '(none found — see errors above)'}")

    post_results, comment_results = run_arcticshift_retrieval(results_dir, candidates)

    comparison = {
        "config": {
            "days_window": DAYS_WINDOW,
            "test_queries": TEST_QUERIES,
            "known_subreddits_tested": candidates or KNOWN_SUBREDDITS,
            "discovery_source_used": discovery_source_used,
        },
        "pullpush_discovery": summarize(pullpush_results),
        "reddit_official_discovery": summarize(reddit_official_results) if reddit_official_results else "skipped (no credentials configured)",
        "arcticshift_fetch_posts": summarize(post_results),
        "arcticshift_fetch_comments": summarize(comment_results),
        "top_candidate_subreddits": candidates,
    }

    save_json(results_dir, "comparison_summary.json", comparison)

    print("\n=== SUMMARY ===")
    print(json.dumps(comparison, indent=2, default=str))
    print(f"\nRaw + summary JSON written to: {results_dir}/")


if __name__ == "__main__":
    main()
