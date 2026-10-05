"""
Merge human annotations from an older annotation_pool.csv snapshot
into the current, freshly-rebuilt pool.

Use this once whenever you have a hand-labeled CSV (from a previous
run, possibly with different thread IDs or a different community mix)
that you want to fold into the latest pool produced by build_pool.py.

Usage:

    python merge_annotations.py OLD_ANNOTATIONS.csv

Optional:

    python merge_annotations.py OLD_ANNOTATIONS.csv --pool path/to/annotation_pool.csv

What it does:
  1. Backs up the current pool.
  2. Loads old labels keyed by (case_id, item_type, item_id).
  3. For every row in the current pool, if it has no relevance/evidence/
     notes yet and a match exists in the old file, copies the old
     labels in.
  4. Never overwrites a label that's already present in the current
     pool (current pool wins if both are labeled).
  5. Reports:
       - how many rows were filled in from the old file
       - how many old-file rows had no matching row in the current
         pool (e.g. threads that were dropped in the new retrieval
         run, or renamed communities) -- written to a leftovers CSV
         so nothing is silently lost.
"""

import argparse
import csv
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_POOL_PATH = BASE_DIR / "data" / "pools" / "latest" / "annotation_pool.csv"

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

LABEL_COLUMNS = ("relevance", "evidence", "notes")


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def row_key(row: dict) -> tuple[str, str, str]:
    return (
        row.get("case_id", "").strip(),
        row.get("item_type", "").strip(),
        row.get("item_id", "").strip(),
    )


def has_any_label(row: dict) -> bool:
    return any(str(row.get(col, "")).strip() for col in LABEL_COLUMNS)


def sanitize_row(row: dict) -> dict:
    """
    Old annotation files may carry extra columns (e.g. post_id,
    sources, source_records from an earlier pipeline version).
    csv.DictWriter refuses to write a dict with keys outside its
    declared fieldnames, so trim every row down to exactly the
    columns the current pool format uses, filling in "" for any
    that are missing.
    """

    return {col: row.get(col, "") for col in FIELDNAMES}


def backup(path: Path, label: str):
    if not path.exists():
        return None

    backup_dir = path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"{path.stem}.{label}.{timestamp}.csv"

    shutil.copy2(path, backup_path)
    print(f"Backed up {path.name} -> {backup_path}")

    return backup_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "old_annotations",
        type=Path,
        help="Path to the older, hand-annotated CSV to pull labels from.",
    )
    parser.add_argument(
        "--pool",
        type=Path,
        default=DEFAULT_POOL_PATH,
        help=f"Path to the current pool to merge into (default: {DEFAULT_POOL_PATH})",
    )
    args = parser.parse_args()

    if not args.old_annotations.exists():
        sys.exit(f"Old annotations file not found: {args.old_annotations}")

    if not args.pool.exists():
        sys.exit(
            f"Current pool not found: {args.pool}\n"
            f"Run build_pool.py first to generate it."
        )

    old_rows = read_csv_rows(args.old_annotations)
    current_rows = read_csv_rows(args.pool)

    old_by_key: dict[tuple[str, str, str], dict] = {}
    old_labeled_count = 0

    for row in old_rows:
        key = row_key(row)

        if not all(key):
            continue

        if has_any_label(row):
            old_by_key[key] = row
            old_labeled_count += 1

    print(f"Old file: {len(old_rows)} rows total, {old_labeled_count} labeled.")

    if old_labeled_count == 0:
        print(
            "WARNING: no labeled rows found in the old file "
            "(relevance/evidence/notes are all empty for every row). "
            "Nothing to merge."
        )

    filled_count = 0
    already_labeled_count = 0
    matched_keys: set[tuple[str, str, str]] = set()

    for row in current_rows:
        key = row_key(row)
        old_row = old_by_key.get(key)

        if old_row is None:
            continue

        matched_keys.add(key)

        if has_any_label(row):
            # Current pool already has labels for this item -- don't
            # clobber newer annotation work with older labels.
            already_labeled_count += 1
            continue

        for col in LABEL_COLUMNS:
            row[col] = old_row.get(col, "")

        filled_count += 1

    # Anything in the old file that has no home in the current pool
    # (dropped thread, renamed/removed community, etc.) -- don't lose
    # it silently. Sanitize in case the old file has extra columns
    # (e.g. post_id, sources, source_records from an older pipeline
    # version) that aren't part of the current pool schema.
    leftover_rows = [
        sanitize_row(row)
        for key, row in old_by_key.items()
        if key not in matched_keys
    ]

    backup(args.pool, "pre-merge")

    with args.pool.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(sanitize_row(row) for row in current_rows)

    leftover_path = args.pool.parent / "unmatched_old_annotations.csv"

    if leftover_rows:
        with leftover_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(leftover_rows)

    print()
    print(f"Filled in from old file:        {filled_count}")
    print(f"Already labeled (kept as-is):   {already_labeled_count}")
    print(f"Old labeled rows with no match: {len(leftover_rows)}")

    if leftover_rows:
        print(f"  -> written to {leftover_path}")
        print("  These are old annotations for items that no longer")
        print("  appear in the current pool (e.g. a thread that fell")
        print("  out of the top-N in this run, or a community that")
        print("  was renamed). Review them manually if you want to")
        print("  keep that judgment for a future re-run.")

    print()
    print(f"Updated pool written to: {args.pool}")


if __name__ == "__main__":
    main()