import json
import os
import shlex
import shutil
import subprocess
from typing import Any

from models import CommunityCandidate


class AgentReachSource:
    """
    Agent Reach / OpenCLI Reddit retrieval adapter.

    Agent Reach uses OpenCLI on the local machine to search Reddit.
    The adapter converts returned Reddit posts into CommunityCandidate
    objects by extracting the subreddit from each post.

    Important:
    - source_rank is the rank of the Reddit post within its query.
    - It is NOT a semantic community relevance score.
    - Multiple posts from the same subreddit are retained initially.
    - Community-level deduplication happens later in the pipeline.
    """

    def __init__(self) -> None:
        self.command = os.getenv("AGENT_REACH_COMMAND", "opencli")
        self.timeout = int(os.getenv("AGENT_REACH_TIMEOUT", "120"))

    def discover(self, case: dict[str, Any]) -> list[CommunityCandidate]:
        case_id = case["case_id"]
        queries = case.get("queries", [])

        all_candidates: list[CommunityCandidate] = []

        for query in queries:
            try:
                posts = self._search_reddit(query)

                normalized = self._normalize_posts(
                    posts=posts,
                    query=query,
                    case_id=case_id,
                )

                all_candidates.extend(normalized)

            except Exception as exc:
                print(
                    f"[Agent Reach] Query failed: {query!r}: "
                    f"{type(exc).__name__}: {exc}"
                )

        return all_candidates

    def _search_reddit(self, query: str) -> list[dict[str, Any]]:
        command = self._resolve_command()

        full_command = [
            *command,
            "reddit",
            "search",
            query,
            "-f",
            "json",
        ]

        process = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout,
            check=False,
        )

        if process.returncode != 0:
            stderr = process.stderr.strip()

            raise RuntimeError(
                f"OpenCLI exited with code {process.returncode}"
                + (f": {stderr}" if stderr else "")
            )

        stdout = process.stdout.strip()

        if not stdout:
            return []

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as exc:
            preview = stdout[:500].replace("\n", "\\n")

            raise RuntimeError(
                f"OpenCLI returned non-JSON output: {preview}"
            ) from exc

        return self._extract_posts(data)

    def _resolve_command(self) -> list[str]:
        """
        Resolve the OpenCLI executable.

        On Windows, `opencli` may be exposed as opencli.cmd rather
        than a directly executable binary. shutil.which() handles
        the normal PATH case, but we also explicitly check .cmd.
        """

        parts = shlex.split(self.command, posix=False)

        if not parts:
            raise RuntimeError("AGENT_REACH_COMMAND is empty.")

        executable = parts[0]
        extra_args = parts[1:]

        resolved = shutil.which(executable)

        if resolved:
            return [resolved, *extra_args]

        if os.name == "nt" and not executable.lower().endswith(".cmd"):
            resolved = shutil.which(f"{executable}.cmd")

            if resolved:
                return [resolved, *extra_args]

        raise FileNotFoundError(
            f"Could not find Agent Reach command: {executable!r}. "
            f"Run `where.exe {executable}` to verify it is on PATH."
        )

    @staticmethod
    def _extract_posts(data: Any) -> list[dict[str, Any]]:
        """
        Normalize the different JSON wrapper shapes OpenCLI may return.

        Expected common shape:

        [
            {...},
            {...}
        ]

        Also supports:

        {
            "results": [...]
        }

        {
            "data": [...]
        }

        {
            "data": {
                "results": [...]
            }
        }
        """

        if isinstance(data, list):
            return [
                item
                for item in data
                if isinstance(item, dict)
            ]

        if not isinstance(data, dict):
            return []

        for key in ("results", "posts", "items"):
            value = data.get(key)

            if isinstance(value, list):
                return [
                    item
                    for item in value
                    if isinstance(item, dict)
                ]

        nested_data = data.get("data")

        if isinstance(nested_data, list):
            return [
                item
                for item in nested_data
                if isinstance(item, dict)
            ]

        if isinstance(nested_data, dict):
            return AgentReachSource._extract_posts(nested_data)

        return []

    @staticmethod
    def _normalize_posts(
        posts: list[dict[str, Any]],
        query: str,
        case_id: str,
    ) -> list[CommunityCandidate]:

        candidates: list[CommunityCandidate] = []

        for rank, post in enumerate(posts, start=1):
            subreddit = AgentReachSource._extract_subreddit(post)

            if not subreddit:
                continue

            subreddit = AgentReachSource._clean_subreddit(subreddit)

            if not subreddit:
                continue

            metadata = {
                "query": query,
                "post_id": AgentReachSource._first_value(
                    post,
                    "id",
                    "post_id",
                ),
                "title": AgentReachSource._first_value(
                    post,
                    "title",
                ),
                "url": AgentReachSource._first_value(
                    post,
                    "url",
                    "permalink",
                ),
                "author": AgentReachSource._first_value(
                    post,
                    "author",
                    "author_name",
                ),
                "score": AgentReachSource._first_value(
                    post,
                    "score",
                    "ups",
                ),
                "num_comments": AgentReachSource._first_value(
                    post,
                    "num_comments",
                    "comment_count",
                ),
            }

            candidates.append(
                CommunityCandidate(
                    case_id=case_id,
                    subreddit=subreddit,
                    source="agent_reach",
                    source_rank=rank,
                    metadata=metadata,
                )
            )

        return candidates

    @staticmethod
    def _extract_subreddit(post: dict[str, Any]) -> str | None:
        """
        Extract subreddit from common Reddit/OpenCLI field names.
        """

        value = AgentReachSource._first_value(
            post,
            "subreddit",
            "subreddit_name",
            "community",
            "community_name",
        )

        if value is not None:
            return str(value)

        # Some payloads may put the subreddit inside a nested
        # community object.
        community = post.get("community")

        if isinstance(community, dict):
            value = AgentReachSource._first_value(
                community,
                "name",
                "display_name",
                "subreddit",
            )

            if value is not None:
                return str(value)

        # Last-resort extraction from a Reddit URL.
        url = AgentReachSource._first_value(
            post,
            "url",
            "permalink",
        )

        if isinstance(url, str):
            marker = "/r/"

            if marker in url:
                remainder = url.split(marker, 1)[1]
                subreddit = remainder.split("/", 1)[0]

                if subreddit:
                    return subreddit

        return None

    @staticmethod
    def _clean_subreddit(value: str) -> str:
        """
        Convert variants such as:

            r/studypartner
            /r/studypartner
            StudyPartner

        into:

            studypartner
        """

        value = value.strip()

        if value.startswith("/r/"):
            value = value[3:]

        elif value.startswith("r/"):
            value = value[2:]

        value = value.strip("/").strip()

        return value.lower()

    @staticmethod
    def _first_value(
        data: dict[str, Any],
        *keys: str,
    ) -> Any:
        for key in keys:
            value = data.get(key)

            if value is not None:
                return value

        return None