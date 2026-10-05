import asyncio
import json
import sys


async def main():

    payload = json.loads(
        sys.stdin.read()
    )

    if payload["operation"] != (
        "discover_subreddits"
    ):
        raise ValueError(
            "Unsupported operation"
        )

    queries = payload[
        "queries"
    ]

    limit = payload.get(
        "limit",
        10,
    )

    # -------------------------------------------------
    # MCP CONNECTION
    # -------------------------------------------------
    #
    # Connect your actual Reddit MCP server here.
    #
    # The server's documented flow is:
    #
    # 1. discover_operations()
    # 2. get_operation_schema(
    #        "discover_subreddits"
    #    )
    # 3. execute_operation(
    #        "discover_subreddits",
    #        {...}
    #    )
    #
    # Do NOT replace this with a guessed REST endpoint.
    #
    # -------------------------------------------------

    raise RuntimeError(
        "Configure the actual Reddit MCP transport "
        "for this environment before running."
    )


if __name__ == "__main__":

    asyncio.run(
        main()
    )