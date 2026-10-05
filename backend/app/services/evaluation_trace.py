"""Opt-in, server-side traces for reproducible analyze evaluations."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


TRACE_DIRECTORY_ENV = "COMMUNITY_INTELLIGENCE_TRACE_DIR"


class EvaluationTraceNotConfigured(RuntimeError):
    pass


def evaluation_trace_directory(directory: str | Path | None = None) -> Path:
    """Resolve the configured trace destination, failing before analysis work."""
    target_dir = directory or os.getenv(TRACE_DIRECTORY_ENV)
    if not target_dir:
        raise EvaluationTraceNotConfigured(
            f"Set {TRACE_DIRECTORY_ENV} before requesting capture_trace."
        )
    return Path(target_dir)


def write_evaluation_trace(trace: dict, directory: str | Path | None = None) -> str:
    """Atomically save one opt-in trace and return its opaque ID."""
    target_dir = evaluation_trace_directory(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    trace_id = uuid.uuid4().hex
    output_path = target_dir / f"{trace_id}.json"
    temporary_path = target_dir / f".{trace_id}.tmp"
    trace_payload = dict(trace)
    final_response = trace_payload.get("final_response")
    if isinstance(final_response, dict):
        final_response = dict(final_response)
        final_response["evaluation_trace_id"] = trace_id
        trace_payload["final_response"] = final_response
    payload = {
        **trace_payload,
        "schema_version": 1,
        "trace_id": trace_id,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }

    serialized = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    with temporary_path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.write("\n")
    os.replace(temporary_path, output_path)
    return trace_id
