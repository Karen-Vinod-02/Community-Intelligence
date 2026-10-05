# Production evaluation tracing

Tracing is disabled by default. It does not change discovery, filtering, scoring, thresholds, or ranking order.

## Enable a capture

Configure a server-side directory, then opt in on an individual request:

```powershell
$env:COMMUNITY_INTELLIGENCE_TRACE_DIR = "C:\community-intelligence-traces"
```

```json
{
  "description": "The exact user problem or product description",
  "capture_trace": true
}
```

The API returns `evaluation_trace_id`; the corresponding JSON file is written as `<id>.json` in the configured directory. A capture request returns HTTP 503 before discovery/retrieval if no trace directory is configured. If ranking raises after inputs were captured, the trace records the pipeline error and the original exception is re-raised. Requests without `capture_trace: true` do not write a trace.

## Captured fields

The trace includes the exact description and extracted queries; provider names; per-query discovery results, errors, and elapsed time; bounded communities; per-community retrieval status, elapsed time, and full returned Reddit post records; usable posts before `posts_per_community` truncation; the exact post inputs passed to the community filter and ranker; filter scores and order; all ranker per-post scores, evidence signals, candidate tier counts, ranker configuration, stage elapsed times, ranked outputs; and the final API response.

The trace contains user descriptions and Reddit post text. Enable capture only for controlled evaluation requests and protect the configured directory according to your data-handling requirements. Traces are not automatically uploaded or sent to an external service.

## Evaluation use

Keep the trace JSON with the source revision and annotation snapshot. Candidate pools, ranker inputs, scores, model identity (when exposed by the embedding model), statuses, and final order can then be compared with a popularity baseline on the same candidates. The trace records the actual retrieval cap and selected ranker input; it does not recover posts beyond what the configured Reddit source returned.
