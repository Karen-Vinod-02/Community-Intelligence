import csv
from pathlib import Path

# File location: <project_root>/data/annotations/annotate.py
ROOT = Path(__file__).resolve().parents[2]

POOL_DIR = ROOT / "data" / "pools" / "latest"
POOL_FILES = [
    POOL_DIR / "annotation_pool.csv",
    POOL_DIR / "agent_reach_annotation_pool_screened.csv",
]

OUTPUT_DIR = ROOT / "data" / "annotations" / "latest"
OUTPUT_FILE = OUTPUT_DIR / "annotations.csv"

# False: relevance/evidence/notes already stored inside the pool files are
#        ignored, so those items are shown to you for judging.
# True:  labels inside the pool files are imported as final judgments.
TRUST_POOL_LABELS = True

FIELDS = [
    "case_id", "item_type", "item_id", "subreddit", "post_id", "title",
    "text", "url", "sources", "relevance", "evidence", "notes",
]
JUDGMENT_FIELDS = ("relevance", "evidence", "notes")
VALID_RELEVANCE = {"0", "1", "2", "3"}


def clean_row(row):
    return {f: (row.get(f) or "") for f in FIELDS}


def item_key(row):
    return (
        row.get("case_id", "").strip(),
        row.get("item_type", "").strip(),
        row.get("item_id", "").strip(),
    )


def has_key(row):
    _, item_type, item_id = item_key(row)
    return bool(item_type and item_id)


def has_relevance(row):
    return row.get("relevance", "").strip() in VALID_RELEVANCE


def load_csv(path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_pool():
    rows = []
    for path in POOL_FILES:
        if not path.exists():
            print(f"Warning: pool file missing, skipped: {path}")
            continue
        loaded = load_csv(path)
        print(f"Loaded {len(loaded)} rows from {path.name}")
        rows.extend(loaded)
    return rows


def save(rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ask_int(prompt, minimum, maximum):
    while True:
        value = input(prompt).strip()
        try:
            number = int(value)
            if minimum <= number <= maximum:
                return number
        except ValueError:
            pass
        print(f"Enter an integer from {minimum} to {maximum}.")


def merge_pool_with_existing(pool_rows, existing_rows):
    """Keep every existing annotation; add pool items; refresh metadata."""
    merged = []
    positions = {}

    for existing in existing_rows:
        row = clean_row(existing)
        key = item_key(row)
        if has_key(row) and key not in positions:
            positions[key] = len(merged)
        merged.append(row)  # malformed/duplicate legacy rows are kept too

    seen = set()
    for pool_row in pool_rows:
        row = clean_row(pool_row)
        key = item_key(row)

        if not has_key(row):
            print("Warning: skipping pool row without item_type/item_id.")
            continue
        if key in seen:
            print(f"Warning: duplicate pool item skipped: {key}")
            continue
        seen.add(key)

        if key in positions:
            previous = merged[positions[key]]
            for f in FIELDS:
                if f not in JUDGMENT_FIELDS and not row[f] and previous[f]:
                    row[f] = previous[f]
            if has_relevance(previous):
                for f in JUDGMENT_FIELDS:
                    row[f] = previous[f]
            elif not TRUST_POOL_LABELS:
                for f in JUDGMENT_FIELDS:
                    row[f] = ""
            merged[positions[key]] = row
        else:
            if not TRUST_POOL_LABELS:
                for f in JUDGMENT_FIELDS:
                    row[f] = ""
            positions[key] = len(merged)
            merged.append(row)

    return merged


def main():
    pool_rows = load_pool()
    if not pool_rows:
        raise FileNotFoundError("No annotation pool rows found.")

    existing_rows = load_csv(OUTPUT_FILE)
    judged_before = sum(1 for r in existing_rows if has_relevance(r))

    rows = merge_pool_with_existing(pool_rows, existing_rows)

    judged_after = sum(1 for r in rows if has_relevance(r))
    if judged_after < judged_before:
        raise RuntimeError(
            f"Judged count would drop ({judged_before} -> {judged_after}). "
            "Nothing was saved."
        )

    pool_keys = {item_key(clean_row(r)) for r in pool_rows if has_key(r)}
    todo = [r for r in rows if item_key(r) in pool_keys and not has_relevance(r)]

    print(f"\nExisting judged rows: {judged_before}")
    print(f"Items to annotate now: {len(todo)}")

    for i, row in enumerate(todo, start=1):
        print()
        print("=" * 80)
        print(f"[{i}/{len(todo)}]  CASE: {row['case_id']}")
        print(f"TYPE: {row['item_type']}")

        subreddit = row["subreddit"].strip()
        print(f"SUBREDDIT: r/{subreddit}" if subreddit else "SUBREDDIT: Not specified")

        if row["title"].strip():
            print(f"\nTITLE:\n{row['title'].strip()}")

        print(f"\nTEXT:\n{row['text'].strip()[:4000]}")
        print(f"\nURL:\n{row['url']}")
        # 'sources' is deliberately not shown, to keep judging blind.

        relevance = ask_int(
            "\nRelevance (0=irrelevant, 1=partially relevant, "
            "2=relevant, 3=highly relevant): ",
            0,
            3,
        )
        row["relevance"] = str(relevance)
        row["evidence"] = input("Evidence (short one-line justification): ").strip()
        row["notes"] = input("Notes: ").strip()

        save(rows)
        print("Saved.")

    save(rows)

    judged = sum(1 for r in rows if has_relevance(r))
    print("\nAnnotation pass complete.")
    print(f"Total rows: {len(rows)}")
    print(f"Rows with relevance judgments: {judged}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()