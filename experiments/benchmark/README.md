# PullPush vs Arctic Shift Benchmark
This benchmark evaluates Reddit data sources before committing to a Reddit data architecture.

It covers four questions:
1. Can Reddit-wide search reliably find relevant subreddits for a keyword?
2. Can Arctic Shift retrieve posts/comments for those subreddits?
3. Can results be filtered to the ~last 90 days?
4. What are the average response times?

## Setup
1. Install Dependencies
```
cd experiments
pip install -r requirements.txt
```
2. Running the Benchmark
```
cd benchmark
python benchmark.py
```
This will:
- Run each query in config.TEST_QUERIES against the configured discovery sources, then rank the subreddits that show up most often in the results.
- Take the top subreddits found and fetch recent posts from them via Arctic Shift, then fetch one comment tree per subreddit to verify the full discover → retrieve → comments pipeline.
- Save every raw response and a rolled-up `comparison_summary.json` to
  `results/`.

No API keys needed for PullPush or Arctic Shift. The Reddit Official API benchmark requires credentials.

Runtime is a few minutes. PullPush calls are deliberately spaced ~2.5s apart to respect its ~30 req/min ceiling.

## Interpreting `results/comparison_summary.json`

- `success_rate`: indicates whether benchmark calls completed successfully. Interpret alongside HTTP status and errors. A consistently low rate may indicate that PullPush is unreliable as the primary discovery source and requires stronger caching/retry logic.
- `avg_latency_seconds` / `p95_latency_seconds`: Useful for evaluating latency and whether aggressive caching is needed.
- `top_candidate_subreddits`: Sanity-check these manually. If a clear query (e.g."social media scheduling")  produces irrelevant subreddits, the match-count ranking heuristic may need additional signals such as score or recency
- `arcticshift_fetch_posts.success_rate`: Measures whether Arctic Shift can reliably retrieve posts once a subreddit is known. This is the relevant retrieval metric for the proposed architecture.


This benchmark is intentionally standalone and is not meant to be imported by the FastAPI app.
