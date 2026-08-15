# backend/app/services/query_extraction.py
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def extract_search_queries(description: str, max_queries: int = 3) -> list[str]:
    clauses = re.split(r"[,.;]|\band\b|\bwho\b|\bwhere\b", description)

    queries = []
    for clause in clauses:
        words = [
            w for w in re.findall(r"[a-zA-Z]+", clause.lower())
            if w not in ENGLISH_STOP_WORDS
        ]
        if 2 <= len(words) <= 6:
            queries.append(" ".join(words))

    if not queries:
        words = [
            w for w in re.findall(r"[a-zA-Z]+", description.lower())
            if w not in ENGLISH_STOP_WORDS
        ]
        queries = [" ".join(words[:6])]

    return queries[:max_queries]