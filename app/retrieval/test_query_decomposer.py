from app.retrieval.query_decomposer import (decompose_query,)


def main():

    queries = [
        (
            "Is this candidate a good fit for "
            "the AI Engineer role?"
        ),
        (
            "The candidate lacks AWS, but are "
            "their projects and experience still "
            "strong enough for this role?"
        ),
        (
            "Is this job worth applying to, "
            "and does the company show any "
            "recruitment warning signs?"
        ),
    ]

    for query in queries:

        subqueries = decompose_query(query)

        print("\n" + "=" * 80)

        print("ORIGINAL")
        print(query)

        print("\nSUBQUERIES")

        for i, subquery in enumerate(subqueries, start=1,):
            print(
                f"{i}. [{subquery.source_type.upper()}] "
                f"{subquery.query}"
            )


if __name__ == "__main__":
    main()