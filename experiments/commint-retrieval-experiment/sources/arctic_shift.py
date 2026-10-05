import time
from datetime import datetime, timedelta, timezone

import requests

from config import (
    ARCTIC_SHIFT_BASE,
    ARCTIC_MAX_RETRIES,
    ARCTIC_COMMENT_LIMIT,
    REQUEST_TIMEOUT,
)

from models import ThreadResult
from sources.base import ThreadRetrievalSource


class ArcticShiftSource(ThreadRetrievalSource):

    name = "arctic_shift"

    # Arctic Shift is currently returning 422 for
    # upstream timeout/rate-pressure conditions.
    # Keep requests deliberately spaced during the pilot.
    REQUEST_DELAY_SECONDS = 2

    # Do not let one case spend many minutes retrying
    # the same failing request.
    MAX_RETRY_WAIT_SECONDS = 15

    def __init__(self):

        self.session = requests.Session()

        self.session.headers.update(
            {
                "User-Agent":
                    "CommInt-Retrieval-Experiment/0.2"
            }
        )

    def _get(
        self,
        endpoint: str,
        params: dict,
    ):
        """
        Perform one GET request against Arctic Shift.

        Retryable:
            - 429 rate limit
            - 422 upstream timeout/rate pressure
            - 5xx server errors
            - network errors
            - timeouts

        Non-retryable:
            - other 4xx client errors

        The important distinction is that Arctic Shift
        currently appears to use HTTP 422 for upstream
        timeout conditions, so 422 is handled as retryable.
        """

        url = (
            f"{ARCTIC_SHIFT_BASE}"
            f"/{endpoint}"
        )

        last_error = None

        for attempt in range(
            ARCTIC_MAX_RETRIES
        ):

            # Deliberately space requests.
            if attempt > 0:
                time.sleep(
                    self.REQUEST_DELAY_SECONDS
                )

            try:

                response = self.session.get(
                    url,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )

            except (
                requests.Timeout,
                requests.ConnectionError,
            ) as exc:

                last_error = exc

                if attempt == (
                    ARCTIC_MAX_RETRIES - 1
                ):
                    raise RuntimeError(
                        "Arctic Shift network request "
                        "failed after "
                        f"{ARCTIC_MAX_RETRIES} attempts: "
                        f"{exc}"
                    ) from exc

                wait = min(
                    2 ** attempt,
                    self.MAX_RETRY_WAIT_SECONDS,
                )

                print(
                    "  Arctic Shift network error: "
                    f"{exc}"
                )

                print(
                    f"  Retrying in {wait}s..."
                )

                time.sleep(wait)

                continue

            status = response.status_code

            # -------------------------------------------------
            # SUCCESS
            # -------------------------------------------------

            if 200 <= status < 300:

                try:

                    return response.json()

                except ValueError as exc:

                    raise RuntimeError(
                        "Arctic Shift returned "
                        "a non-JSON response for "
                        f"{url}"
                    ) from exc

            # -------------------------------------------------
            # RATE LIMIT
            # -------------------------------------------------

            if status == 429:

                last_error = RuntimeError(
                    "Arctic Shift HTTP 429"
                )

                if attempt == (
                    ARCTIC_MAX_RETRIES - 1
                ):
                    raise RuntimeError(
                        "Arctic Shift rate limit "
                        "persisted after "
                        f"{ARCTIC_MAX_RETRIES} attempts."
                    )

                wait = self._rate_limit_wait(
                    response,
                    attempt,
                )

                print(
                    "  Arctic Shift 429; "
                    f"waiting {wait}s"
                )

                time.sleep(wait)

                continue

            # -------------------------------------------------
            # 422 TIMEOUT / RATE PRESSURE
            # -------------------------------------------------

            if status == 422:

                body = (
                    response.text.strip()
                )

                last_error = RuntimeError(
                    "Arctic Shift HTTP 422: "
                    f"{body[:1000]}"
                )

                # Arctic Shift currently returns:
                #
                # {"data":null,"error":
                #  "Timeout. Maybe slow down a bit"}
                #
                # Treat this particular response as
                # retryable rather than a malformed request.

                is_timeout_response = (
                    "timeout" in body.lower()
                    or "slow down" in body.lower()
                )

                if not is_timeout_response:

                    raise RuntimeError(
                        "Arctic Shift HTTP 422 "
                        "with unexpected response: "
                        f"{body[:1000]}"
                    )

                if attempt == (
                    ARCTIC_MAX_RETRIES - 1
                ):

                    raise RuntimeError(
                        "Arctic Shift HTTP 422 "
                        "timeout persisted after "
                        f"{ARCTIC_MAX_RETRIES} attempts: "
                        f"{body[:500]}"
                    )

                wait = min(
                    5 * (attempt + 1),
                    self.MAX_RETRY_WAIT_SECONDS,
                )

                print(
                    "  Arctic Shift 422 "
                    "(upstream timeout/rate pressure); "
                    f"waiting {wait}s"
                )

                time.sleep(wait)

                continue

            # -------------------------------------------------
            # OTHER 4XX
            # -------------------------------------------------

            if 400 <= status < 500:

                body = (
                    response.text.strip()
                )

                raise RuntimeError(
                    "Arctic Shift HTTP "
                    f"{status} for "
                    f"{endpoint}. "
                    f"Response: {body[:1000]}"
                )

            # -------------------------------------------------
            # 5XX
            # -------------------------------------------------

            if 500 <= status < 600:

                body = (
                    response.text.strip()
                )

                last_error = RuntimeError(
                    "Arctic Shift HTTP "
                    f"{status}: "
                    f"{body[:500]}"
                )

                if attempt == (
                    ARCTIC_MAX_RETRIES - 1
                ):
                    raise last_error

                wait = min(
                    2 ** attempt,
                    self.MAX_RETRY_WAIT_SECONDS,
                )

                print(
                    f"  Arctic Shift HTTP "
                    f"{status}; "
                    f"retrying in {wait}s"
                )

                time.sleep(wait)

                continue

            raise RuntimeError(
                "Unexpected Arctic Shift "
                f"HTTP status {status}"
            )

        if last_error:
            raise last_error

        raise RuntimeError(
            "Arctic Shift request failed "
            "without a recorded error."
        )

    @staticmethod
    def _rate_limit_wait(
        response,
        attempt: int,
    ):
        """
        Determine a reasonable wait time for HTTP 429.

        Prefer an explicit rate-limit reset timestamp
        when available. Otherwise use bounded exponential
        backoff.

        The wait is intentionally capped because this is
        a pilot experiment, not a long-running crawler.
        """

        headers = response.headers

        for header_name in (
            "X-RateLimit-Reset",
            "X-RateLimit-Reset-At",
        ):

            value = headers.get(
                header_name
            )

            if not value:
                continue

            try:

                reset_timestamp = float(
                    value
                )

                wait = (
                    reset_timestamp
                    - time.time()
                )

                if wait > 0:

                    return max(
                        1,
                        min(
                            int(wait) + 1,
                            30,
                        ),
                    )

            except (
                TypeError,
                ValueError,
            ):

                pass

        return min(
            5 * (attempt + 1),
            30,
        )

    @staticmethod
    def _after_date(
        days: int,
    ):

        dt = (
            datetime.now(
                timezone.utc
            )
            - timedelta(
                days=days
            )
        )

        return dt.strftime(
            "%Y-%m-%d"
        )

    def search_posts(
        self,
        subreddit: str,
        query: str,
        days: int,
        limit: int,
    ):
        """
        Search posts inside one subreddit.
        """

        result = self._get(
            "posts/search",
            {
                "subreddit": subreddit,
                "query": query,
                "after": self._after_date(
                    days
                ),
                "sort": "desc",
                "limit": limit,
                "fields": (
                    "id,title,selftext,"
                    "author,created_utc,"
                    "score,num_comments"
                ),
            },
        )

        if not isinstance(
            result,
            dict,
        ):

            raise RuntimeError(
                "Unexpected Arctic Shift "
                "posts/search response type: "
                f"{type(result).__name__}"
            )

        data = result.get(
            "data",
            [],
        )

        if data is None:
            return []

        if not isinstance(
            data,
            list,
        ):

            raise RuntimeError(
                "Unexpected Arctic Shift "
                "posts/search data type: "
                f"{type(data).__name__}"
            )

        return data

    def fetch_comments(
        self,
        post_id: str,
    ):
        """
        Retrieve the comment tree for a Reddit post.

        API failures are raised rather than silently
        converted into an empty comment list.
        """

        data = self._get(
            "comments/tree",
            {
                "link_id": f"t3_{post_id}",
                "limit": ARCTIC_COMMENT_LIMIT,
            },
        )

        output = []

        def walk(node):

            if isinstance(
                node,
                list,
            ):

                for child in node:
                    walk(child)

                return

            if not isinstance(
                node,
                dict,
            ):
                return

            if node.get("kind") == "more":
                return

            body = (
                node.get("body")
                or ""
            ).strip()

            if body and body not in {
                "[deleted]",
                "[removed]",
            }:

                output.append(
                    {
                        "id": node.get(
                            "id"
                        ),
                        "body": body,
                        "author": node.get(
                            "author"
                        ),
                        "score": node.get(
                            "score",
                            0,
                        ),
                        "created_at":
                            node.get(
                                "created_utc"
                            ),
                        "parent_id":
                            node.get(
                                "parent_id"
                            ),
                    }
                )

            replies = node.get(
                "replies"
            )

            if replies:
                walk(replies)

            nested_data = node.get(
                "data"
            )

            if (
                nested_data
                and nested_data is not node
            ):
                walk(nested_data)

        payload = data

        if isinstance(
            data,
            dict,
        ):

            if "data" in data:
                payload = data["data"]

            elif "comments" in data:
                payload = data["comments"]

        walk(payload)

        # Remove duplicate comment IDs.
        seen = set()
        unique_output = []

        for comment in output:

            comment_id = comment.get(
                "id"
            )

            if (
                comment_id
                and comment_id in seen
            ):
                continue

            if comment_id:
                seen.add(
                    comment_id
                )

            unique_output.append(
                comment
            )

        unique_output.sort(
            key=lambda x: (
                x.get(
                    "score",
                    0,
                )
                or 0
            ),
            reverse=True,
        )

        return unique_output

    def retrieve_threads(
        self,
        case: dict,
        subreddit: str,
    ):
        """
        Retrieve candidate threads from one subreddit.

        For the pilot, thread_queries can be supplied
        separately from the community-discovery queries.

        Example:

            "thread_queries": [
                "study accountability"
            ]

        If thread_queries is absent, the first discovery
        query is used.

        This prevents a 5-query community discovery stage
        from automatically becoming 5 expensive Arctic
        Shift searches per community.
        """

        subreddit = (
            subreddit
            .removeprefix("r/")
            .strip()
        )

        thread_queries = case.get(
            "thread_queries"
        )

        if not thread_queries:

            discovery_queries = case.get(
                "queries",
                [],
            )

            thread_queries = (
                discovery_queries[:1]
                if discovery_queries
                else []
            )

        candidates = {}

        for query_index, query in enumerate(
            thread_queries
        ):

            query = str(
                query
            ).strip()

            if not query:
                continue

            # Space out separate searches.
            if query_index > 0:
                time.sleep(
                    self.REQUEST_DELAY_SECONDS
                )

            try:

                posts = self.search_posts(
                    subreddit=subreddit,
                    query=query,
                    days=case[
                        "time_window_days"
                    ],
                    limit=case[
                        "candidate_posts_per_community"
                    ],
                )

            except Exception as exc:

                print(
                    f"  Search failed "
                    f"r/{subreddit} "
                    f"'{query}': {exc}"
                )

                continue

            for post in posts:

                if not isinstance(
                    post,
                    dict,
                ):
                    continue

                post_id = post.get(
                    "id"
                )

                if not post_id:
                    continue

                if post_id not in candidates:

                    candidates[
                        post_id
                    ] = {
                        **post,
                        "_query": query,
                    }

        threads = []

        for post in candidates.values():

            try:

                score = float(
                    post.get(
                        "score",
                        0,
                    )
                    or 0
                )

            except (
                TypeError,
                ValueError,
            ):

                score = 0.0

            try:

                num_comments = int(
                    post.get(
                        "num_comments",
                        0,
                    )
                    or 0
                )

            except (
                TypeError,
                ValueError,
            ):

                num_comments = 0

            post_id = str(
                post["id"]
            )

            threads.append(
                ThreadResult(
                    case_id=case[
                        "case_id"
                    ],
                    subreddit=subreddit.lower(),
                    post_id=post_id,
                    source=self.name,
                    title=(
                        post.get(
                            "title",
                            "",
                        )
                        or ""
                    ),
                    text=(
                        post.get(
                            "selftext",
                            "",
                        )
                        or ""
                    ),
                    url=(
                        "https://www.reddit.com"
                        f"/r/{subreddit}"
                        f"/comments/"
                        f"{post_id}/"
                    ),
                    author=post.get(
                        "author"
                    ),
                    created_at=post.get(
                        "created_utc"
                    ),
                    score=score,
                    num_comments=num_comments,
                    query=post[
                        "_query"
                    ],
                )
            )

        # Baseline thread ordering.
        #
        # This is not the final CommInt relevance
        # ranking. It simply provides deterministic
        # candidates for the pilot.
        threads.sort(
            key=lambda x: (
                x.num_comments,
                x.score,
            ),
            reverse=True,
        )

        return threads