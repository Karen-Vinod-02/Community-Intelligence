import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class CallResult:
    provider: str
    call_type: str          # "discover" | "fetch_posts" | "fetch_comments"
    query_or_target: str
    timestamp: float
    latency_seconds: float
    http_status: Optional[int]
    ok: bool
    num_results: int
    bytes_returned: int
    error: Optional[str] = None
    subreddits_seen: list = field(default_factory=list)
    raw_sample: Any = None  # first 3 results only, for spot-checking

    def to_dict(self):
        return asdict(self)


class Timer:
    """Context manager that records wall-clock latency in seconds."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start


def summarize(results: list[CallResult]) -> dict:
    if not results:
        return {}

    ok_results = [r for r in results if r.ok]
    latencies = [r.latency_seconds for r in ok_results]
    counts = [r.num_results for r in ok_results]

    def pct(n):
        if not latencies:
            return None
        s = sorted(latencies)
        idx = min(len(s) - 1, int(len(s) * n))
        return round(s[idx], 3)

    return {
        "total_calls": len(results),
        "successful_calls": len(ok_results),
        "failed_calls": len(results) - len(ok_results),
        "success_rate": round(len(ok_results) / len(results), 3) if results else 0,
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "min_latency_seconds": round(min(latencies), 3) if latencies else None,
        "max_latency_seconds": round(max(latencies), 3) if latencies else None,
        "p50_latency_seconds": pct(0.5),
        "p95_latency_seconds": pct(0.95),
        "avg_results_returned": round(sum(counts) / len(counts), 1) if counts else None,
        "total_results_returned": sum(counts) if counts else 0,
    }
