import math


def ndcg_at_k(
    ranked_labels: list[int],
    k: int,
) -> float:

    actual = ranked_labels[:k]

    ideal = sorted(
        ranked_labels,
        reverse=True,
    )[:k]

    def dcg(labels):

        total = 0.0

        for index, relevance in enumerate(
            labels,
            start=1,
        ):

            total += (
                (2 ** relevance - 1)
                / math.log2(index + 1)
            )

        return total

    ideal_score = dcg(ideal)

    if ideal_score == 0:
        return 0.0

    return (
        dcg(actual)
        / ideal_score
    )


def reciprocal_rank(
    ranked_labels: list[int],
) -> float:

    for index, label in enumerate(
        ranked_labels,
        start=1,
    ):

        if label > 0:
            return 1.0 / index

    return 0.0