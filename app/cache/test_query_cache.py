from app.retrieval.query_rewriter import (rewrite_query,)


def main():
    query = (
        "Does this candidate actually "
        "have enough backend stuff?"
    )

    print("FIRST CALL")
    result_1 = rewrite_query(query)
    print(result_1)

    print("\nSECOND CALL")
    result_2 = rewrite_query(query)
    print(result_2)

    print(
        "\nSame result:",
        result_1 == result_2
    )


if __name__ == "__main__":
    main()