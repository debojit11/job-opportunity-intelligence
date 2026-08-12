from app.agents.job_rag_agent import agent


def main():
    queries = [
        "Does the candidate have enough Docker experience?",
        "Are the candidate's projects relevant to this role?",
        "Does the company evidence show recruitment warning signs?",
    ]

    for query in queries:

        print("\n" + "=" * 80)
        print("QUERY")
        print(query)
        print("=" * 80)

        result = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        })

        print("\nMESSAGE TRACE")
        print("-" * 80)

        for i, message in enumerate(
            result["messages"],
            start=1,
        ):
            print(
                f"\n{i}. {type(message).__name__}"
            )

            # Show tool calls made by the model
            tool_calls = getattr(
                message,
                "tool_calls",
                None,
            )

            if tool_calls:
                print("TOOL CALLS:")

                for call in tool_calls:
                    print(
                        f"  - {call.get('name')}"
                    )
                    print(
                        f"    args={call.get('args')}"
                    )

            # Show tool result
            if type(message).__name__ == "ToolMessage":
                print(
                    f"TOOL: "
                    f"{getattr(message, 'name', None)}"
                )

                print(
                    f"RESULT:\n{message.content}"
                )

        final_message = result["messages"][-1]

        print("\nFINAL ANSWER")
        print("-" * 80)

        print(final_message.content)


if __name__ == "__main__":
    main()