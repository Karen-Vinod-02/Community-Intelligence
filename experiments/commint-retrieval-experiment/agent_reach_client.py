import json
import subprocess
import sys


def main():

    payload = json.loads(
        sys.stdin.read()
    )

    queries = payload.get(
        "queries",
        [],
    )

    limit = payload.get(
        "limit",
        10,
    )

    results = []

    for query in queries:

        command = [
            "rdt",
            "search",
            query,
        ]

        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        if process.returncode != 0:
            continue

        # The exact rdt output format can vary.
        #
        # Inspect:
        #
        #     rdt search "study partner"
        #
        # on your installed version first.
        #
        # Then normalize its actual JSON output here.

        try:

            data = json.loads(
                process.stdout
            )

        except json.JSONDecodeError:

            continue

        if isinstance(
            data,
            list,
        ):

            results.extend(
                data
            )

        elif isinstance(
            data,
            dict,
        ):

            results.extend(
                data.get(
                    "results",
                    []
                )
            )

        if len(results) >= limit:
            break

    print(
        json.dumps(
            {
                "results": results[
                    :limit
                ]
            }
        )
    )


if __name__ == "__main__":
    main()