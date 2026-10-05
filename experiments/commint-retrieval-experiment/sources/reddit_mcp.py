import asyncio
import json
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from models import CommunityCandidate
from sources.base import CommunityDiscoverySource


class RedditMCPSource(CommunityDiscoverySource):
    """
    Reddit MCP community-discovery source.

    Authentication is handled by mcp-remote.

    Architecture:

        Python
            |
            | MCP over STDIO
            v
        mcp-remote
            |
            | OAuth + Streamable HTTP
            v
        mcp.dialog.tools
    """

    name = "reddit_mcp"

    def __init__(self):
        self.server_url = os.getenv(
            "REDDIT_MCP_URL",
            "https://mcp.dialog.tools/mcp",
        )

        self.auth_timeout = os.getenv(
            "REDDIT_MCP_AUTH_TIMEOUT",
            "120",
        )

    def discover(self, case):
        return asyncio.run(
            self._discover(case)
        )

    async def _discover(self, case):
        print(
            "[reddit_mcp] Starting mcp-remote bridge..."
        )

        server_params = self._build_server_params()

        all_candidates = []

        async with stdio_client(
            server_params
        ) as (read_stream, write_stream):

            async with ClientSession(
                read_stream,
                write_stream,
            ) as session:

                print(
                    "[reddit_mcp] Initializing MCP session..."
                )

                await session.initialize()

                print(
                    "[reddit_mcp] MCP session established."
                )

                print(
                    "[reddit_mcp] Discovering operations..."
                )

                await session.call_tool(
                    "discover_operations",
                    {},
                )

                print(
                    "[reddit_mcp] Operations discovered."
                )

                print(
                    "[reddit_mcp] Inspecting "
                    "discover_subreddits schema..."
                )

                await session.call_tool(
                    "get_operation_schema",
                    {
                        "operation_id": "discover_subreddits",
                        "include_examples": True,
                    },
                )

                print(
                    "[reddit_mcp] "
                    "discover_subreddits schema retrieved."
                )

                queries = case.get(
                    "queries",
                    [],
                )

                limit = case.get(
                    "top_communities",
                    5,
                )

                for query in queries:
                    print(
                        f"[reddit_mcp] Searching: {query}"
                    )

                    parameters = {
                        "query": query,
                        "limit": limit,
                    }

                    if "mcp_min_confidence" in case:
                        parameters[
                            "min_confidence"
                        ] = case[
                            "mcp_min_confidence"
                        ]

                    result = await session.call_tool(
                        "execute_operation",
                        {
                            "operation_id":
                                "discover_subreddits",
                            "parameters":
                                parameters,
                        },
                    )

                    data = self._extract_result(
                        result
                    )

                    candidates = self._normalize(
                        case=case,
                        query=query,
                        data=data,
                    )

                    print(
                        "[reddit_mcp] "
                        f"Found {len(candidates)} "
                        "communities for query."
                    )

                    # IMPORTANT:
                    # Preserve every query-level discovery result.
                    #
                    # Do NOT deduplicate here.
                    # The downstream normalization stage needs all
                    # query/source records to calculate query coverage
                    # and preserve discovery evidence.
                    all_candidates.extend(
                        candidates
                    )

        print(
            "[reddit_mcp] Total raw candidates: "
            f"{len(all_candidates)}"
        )

        # Return all raw discovery records.
        #
        # Deduplication is intentionally handled later by:
        #
        #   evaluation.normalize.deduplicate_communities()
        #
        # This preserves repeated discoveries such as:
        #
        #   r/studying <- "study partner"
        #   r/studying <- "study motivation"
        #
        # allowing query_coverage to be calculated correctly.
        return all_candidates

    def _build_server_params(self):
        """
        Launch mcp-remote as the local STDIO MCP server.

        On Windows, npx.cmd is required when spawning npx
        through a subprocess.
        """

        if os.name == "nt":
            command = "npx.cmd"
        else:
            command = "npx"

        return StdioServerParameters(
            command=command,
            args=[
                "-y",
                "mcp-remote@latest",
                self.server_url,
                "--auth-timeout",
                str(self.auth_timeout),
                "--allow-http",
            ],
            env={
                "PATH": os.environ.get(
                    "PATH",
                    "",
                ),
            },
        )

    @staticmethod
    def _extract_result(result):
        """
        Extract ordinary Python data from an MCP CallToolResult.

        Handles structured content as well as JSON/text content.
        """

        structured = getattr(
            result,
            "structured_content",
            None,
        )

        if structured:
            return structured

        structured = getattr(
            result,
            "structuredContent",
            None,
        )

        if structured:
            return structured

        content = getattr(
            result,
            "content",
            None,
        )

        if content:
            for item in content:
                text = getattr(
                    item,
                    "text",
                    None,
                )

                if text is None:
                    continue

                try:
                    return json.loads(
                        text
                    )

                except json.JSONDecodeError:
                    return text

        return result

    def _normalize(
        self,
        case,
        query,
        data,
    ):
        """
        Convert the Reddit MCP response into
        CommunityCandidate objects.

        Actual MCP schema:

            {
                "success": true,
                "data": {
                    "query": "...",
                    "subreddits": [
                        {
                            "name": "...",
                            "subscribers": ...,
                            "confidence": ...,
                            "distance": ...,
                            "match_tier": "...",
                            "url": "..."
                        }
                    ],
                    "summary": {...},
                    "next_actions": [...]
                }
            }
        """

        if not isinstance(data, dict):
            return []

        payload = data.get(
            "data",
            data,
        )

        if not isinstance(payload, dict):
            return []

        items = payload.get(
            "subreddits",
            [],
        )

        if not isinstance(items, list):
            return []

        output = []

        for index, item in enumerate(
            items,
            start=1,
        ):
            if not isinstance(item, dict):
                continue

            subreddit = (
                item.get("name")
                or ""
            )

            subreddit = (
                str(subreddit)
                .strip()
                .removeprefix("r/")
                .lower()
            )

            if not subreddit:
                continue

            try:
                confidence = float(
                    item.get(
                        "confidence",
                        0.0,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                confidence = 0.0

            try:
                distance = float(
                    item.get(
                        "distance",
                        0.0,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                distance = 0.0

            subscribers = item.get(
                "subscribers",
                0,
            )

            try:
                subscribers = int(
                    subscribers
                )
            except (
                TypeError,
                ValueError,
            ):
                subscribers = 0

            match_tier = str(
                item.get(
                    "match_tier",
                    "",
                )
            )

            url = str(
                item.get(
                    "url",
                    (
                        "https://www.reddit.com/r/"
                        f"{subreddit}/"
                    ),
                )
            )

            output.append(
                CommunityCandidate(
                    case_id=case["case_id"],
                    subreddit=subreddit,
                    source=self.name,
                    source_rank=index,
                    source_score=confidence,
                    name=f"r/{subreddit}",
                    description="",
                    url=url,
                    discovery_reason=(
                        f"MCP semantic discovery; "
                        f"match tier: {match_tier}"
                    ),
                    metadata={
                        "query": query,
                        "confidence": confidence,
                        "distance": distance,
                        "match_tier": match_tier,
                        "subscribers": subscribers,
                        "raw": item,
                    },
                )
            )

        return output