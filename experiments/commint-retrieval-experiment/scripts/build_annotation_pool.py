import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

NORMALIZED_DIR = BASE_DIR / "data" / "normalized" / "latest"
POOL_DIR = BASE_DIR / "data" / "pools" / "latest"
POOL_PATH = POOL_DIR / "annotation_pool.csv"

FIELDNAMES = [
    "case_id",
    "item_type",
    "item_id",
    "subreddit",
    "title",
    "text",
    "url",
    "relevance",
    "evidence",
    "notes",
]

# These columns are considered "annotator-owned". When the pool is
# rebuilt, any existing values in these columns are carried forward
# by (case_id, item_type, item_id) so that re-running retrieval does
# not wipe out work already done in the CSV.
PRESERVED_COLUMNS = ("relevance", "evidence", "notes")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_labels() -> dict[tuple[str, str, str], dict]:
    """
    Read the current annotation_pool.csv (if any) and return a lookup
    of previously-entered annotator columns, keyed by
    (case_id, item_type, item_id).
    """

    if not POOL_PATH.exists():
        return {}

    existing: dict[tuple[str, str, str], dict] = {}

    with POOL_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            key = (
                row.get("case_id", ""),
                row.get("item_type", ""),
                row.get("item_id", ""),
            )

            existing[key] = {
                col: row.get(col, "")
                for col in PRESERVED_COLUMNS
            }

    return existing


def backup_existing_pool():
    """
    Keep a timestamped copy of the previous pool before overwriting it,
    in case label-carryover ever misses something.
    """

    if not POOL_PATH.exists():
        return

    backup_dir = POOL_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"annotation_pool.{timestamp}.csv"

    shutil.copy2(POOL_PATH, backup_path)
    print(f"Backed up previous pool -> {backup_path}")


def write_pool(rows: list[dict]):
    POOL_DIR.mkdir(parents=True, exist_ok=True)

    backup_existing_pool()

    existing_labels = load_existing_labels()

    carried_over = 0

    for row in rows:
        key = (row["case_id"], row["item_type"], row["item_id"])
        old = existing_labels.get(key)

        if old and any(old.get(col) for col in PRESERVED_COLUMNS):
            for col in PRESERVED_COLUMNS:
                row[col] = old.get(col, "")
            carried_over += 1

    with POOL_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Carried over annotations for {carried_over} existing row(s).")


def build_community_rows(case_id: str, communities: dict) -> list[dict]:
    """
    Only MCP community-discovery results enter the community
    annotation pool.

    communities.json is only trusted when it was produced by the
    current pipeline (community_discovery_source == "reddit_mcp").
    Older pipeline versions merged Agent-Reach results into the
    pool directly, which pollutes annotation with irrelevant
    subreddits (e.g. unrelated post-search hits). If an old-format
    file is found, we refuse to silently use it.
    """

    discovery_source = communities.get("community_discovery_source")

    if discovery_source != "reddit_mcp":
        raise ValueError(
            f"{case_id}: communities.json is missing or has an "
            f"unexpected community_discovery_source "
            f"({discovery_source!r}). This usually means the file "
            f"was written by an older version of the pipeline that "
            f"mixed Agent-Reach results into the community pool. "
            f"Delete data/normalized/latest/{case_id}/communities.json "
            f"and re-run retrieval before rebuilding the pool."
        )

    rows = []

    ranked = communities.get("ranked", [])

    for item in ranked:
        subreddit = str(item.get("subreddit", "")).strip().lower()

        if not subreddit:
            continue

        # Defensive: even within a well-formed file, only keep
        # entries actually sourced from reddit_mcp.
        sources = item.get("sources", [])

        if "reddit_mcp" not in sources:
            continue

        rows.append(
            {
                "case_id": case_id,
                "item_type": "community",
                "item_id": subreddit,
                "subreddit": subreddit,
                "title": "",
                "text": "",
                "url": item.get("url", ""),
                "relevance": "",
                "evidence": "",
                "notes": "",
            }
        )

    return rows


def build_thread_rows(case_id: str, threads) -> list[dict]:
    """
    threads.json can be either:

    - a bare list of thread dicts (older format), or
    - a dict shaped like
        {"case_id": ..., "thread_query_count": ..., "selected": [...]}
      (current format, as written by run_experiment.py)

    Handle both.
    """

    if isinstance(threads, dict):
        threads = threads.get("selected", [])

    if not isinstance(threads, list):
        raise ValueError(
            f"Expected threads.json for {case_id} to contain a list "
            f"(either at the top level or under 'selected'), got "
            f"{type(threads).__name__}"
        )

    rows = []

    for item in threads:
        post_id = str(item.get("post_id", "")).strip()

        if not post_id:
            continue

        subreddit = str(item.get("subreddit", "")).strip().lower()

        rows.append(
            {
                "case_id": case_id,
                "item_type": "thread",
                "item_id": post_id,
                "subreddit": subreddit,
                "title": item.get("title", ""),
                "text": item.get("text", ""),
                "url": item.get("url", ""),
                "relevance": "",
                "evidence": "",
                "notes": "",
            }
        )

    return rows


def main():
    all_rows = []

    case_dirs = [
        path
        for path in NORMALIZED_DIR.iterdir()
        if path.is_dir()
    ]

    if not case_dirs:
        raise RuntimeError(
            f"No case directories found under {NORMALIZED_DIR}"
        )

    for case_dir in sorted(case_dirs):
        case_id = case_dir.name

        communities_path = case_dir / "communities.json"
        threads_path = case_dir / "threads.json"

        if communities_path.exists() and communities_path.stat().st_size > 0:
            communities = load_json(communities_path)

            if not isinstance(communities, dict):
                raise ValueError(
                    f"Expected {communities_path} to contain a dict, "
                    f"got {type(communities).__name__}"
                )

            all_rows.extend(
                build_community_rows(case_id, communities)
            )
        else:
            print(f"[{case_id}] no communities.json found, skipping communities.")

        if threads_path.exists() and threads_path.stat().st_size > 0:
            threads = load_json(threads_path)

            all_rows.extend(
                build_thread_rows(case_id, threads)
            )
        else:
            print(f"[{case_id}] no threads.json found, skipping threads.")

    write_pool(all_rows)

    community_count = sum(
        row["item_type"] == "community"
        for row in all_rows
    )

    thread_count = sum(
        row["item_type"] == "thread"
        for row in all_rows
    )

    print(f"Annotation pool created: {POOL_PATH}")
    print(f"Total rows:       {len(all_rows)}")
    print(f"Communities:      {community_count}")
    print(f"Threads:          {thread_count}")


if __name__ == "__main__":
    main()