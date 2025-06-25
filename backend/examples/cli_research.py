import argparse
from langchain_core.messages import HumanMessage
from agent.graph import graph


def main() -> None:
    """Run the research agent from the command line."""
    parser = argparse.ArgumentParser(description="Run the LangGraph research agent")
    parser.add_argument("question", help="Research question")
    parser.add_argument(
        "--initial-queries",
        type=int,
        default=3,
        help="Number of initial search queries",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=2,
        help="Maximum number of research loops",
    )
    parser.add_argument(
        "--model-provider",
        choices=["gemini", "deepseek"],
        default="deepseek",
        help="Model provider to use (gemini or deepseek)",
    )
    parser.add_argument(
        "--no-web-search",
        action="store_true",
        help="Disable web search and use direct answer mode",
    )
    args = parser.parse_args()

    config = {
        "configurable": {
            "model_provider": args.model_provider,
        }
    }

    state = {
        "messages": [HumanMessage(content=args.question)],
        "initial_search_query_count": args.initial_queries,
        "max_research_loops": args.max_loops,
        "enable_web_search": not args.no_web_search,
    }

    print(f"Using {args.model_provider} provider with {'web search' if not args.no_web_search else 'direct answer'} mode")
    
    result = graph.invoke(state, config=config)
    messages = result.get("messages", [])
    if messages:
        print(messages[-1].content)


if __name__ == "__main__":
    main()
