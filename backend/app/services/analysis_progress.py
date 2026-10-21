from threading import Lock
from time import time


_lock = Lock()
_progress: dict[str, dict] = {}
_TTL_SECONDS = 900


def update_analysis_progress(progress_id: str | None, stage: str, percent: int) -> None:
    if not progress_id:
        return
    now = time()
    with _lock:
        expired = [
            key for key, value in _progress.items()
            if now - value["updated_at"] > _TTL_SECONDS
        ]
        for key in expired:
            _progress.pop(key, None)
        _progress[progress_id] = {
            "stage": stage,
            "percent": percent,
            "updated_at": now,
        }


def get_analysis_progress(progress_id: str) -> dict:
    with _lock:
        value = _progress.get(progress_id)
        if value is None:
            return {"stage": "starting", "percent": 0, "complete": False}
        return {
            "stage": value["stage"],
            "percent": value["percent"],
            "complete": value["stage"] in {"complete", "failed"},
        }
