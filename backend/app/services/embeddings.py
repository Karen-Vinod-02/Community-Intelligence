from collections import OrderedDict
from functools import lru_cache
from threading import RLock

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:

    def __init__(self, cache_size: int = 4096):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.cache_size = cache_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = RLock()

    def encode(self, texts: list[str]):
        if not texts:
            return np.empty(
                (0, self.model.get_sentence_embedding_dimension()),
                dtype=np.float32,
            )

        with self._lock:
            missing = list(dict.fromkeys(
                text for text in texts if text not in self._cache
            ))
            if missing:
                vectors = self.model.encode(
                    missing,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                for text, vector in zip(missing, vectors, strict=True):
                    self._cache[text] = vector
                    self._cache.move_to_end(text)

                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)

            rows = []
            for text in texts:
                self._cache.move_to_end(text)
                rows.append(self._cache[text])

            return np.stack(rows)


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()