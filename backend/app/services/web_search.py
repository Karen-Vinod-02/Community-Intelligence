import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from ddgs import DDGS


logger = logging.getLogger(__name__)
DEFAULT_BACKENDS = ("duckduckgo", "bing", "brave")
_SEARCH_EXECUTOR = ThreadPoolExecutor(max_workers=12)


def search_text(
    query: str,
    *,
    max_results: int,
    timeout: int = 5,
    timelimit: str | None = None,
    preferred_backend: str = "duckduckgo",
) -> tuple[list[dict], str | None, Exception | None]:
    """Search independent indexes and return the first non-empty result set."""
    backends = tuple(dict.fromkeys((preferred_backend, *DEFAULT_BACKENDS)))

    def search_backend(backend: str):
        try:
            with DDGS(timeout=timeout) as ddgs:
                results = list(ddgs.text(
                    query,
                    max_results=max_results,
                    backend=backend,
                    timelimit=timelimit,
                ))
        except Exception as exc:
            if "no results found" in str(exc).casefold():
                return backend, [], None
            return backend, [], exc
        return backend, results, None

    failures = []
    successful_backends = []
    futures = [
        _SEARCH_EXECUTOR.submit(search_backend, backend)
        for backend in backends
    ]
    for future in as_completed(futures):
        backend, results, error = future.result()
        if error is not None:
            failures.append(error)
            logger.debug("Indexed search backend %s failed: %s", backend, error)
            continue
        successful_backends.append(backend)
        if results:
            if backend != preferred_backend:
                logger.info(
                    "Indexed search recovered with %s after %s did not return results.",
                    backend,
                    preferred_backend,
                )
            for pending in futures:
                if pending is not future:
                    pending.cancel()
            return results, backend, None

    if successful_backends:
        return [], successful_backends[0], None
    return [], None, failures[-1] if failures else None
