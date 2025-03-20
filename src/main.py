import sys
from langchain_core.runnables import RunnableConfig

# Import the node functions and data structures
from assistant.graph import (
    generate_query, web_research, summarize_sources,
    reflect_on_summary, finalize_summary, route_research
)
from assistant.configuration import Configuration
from assistant.state import SummaryState, SummaryStateInput

def run_research_pipeline(topic: str):
    """
    Manual execution of the 'graph':
      START -> generate_query -> web_research -> summarize_sources -> reflect_on_summary
      Then route to either web_research again or finalize_summary, depending on route_research().
    """
    # Our state holds all the fields (research_topic, running_summary, etc.)
    state = SummaryState(research_topic=topic)

    # Build a RunnableConfig (if needed) for your node calls
    config = RunnableConfig({})

    print(f"[INFO] Starting research on: {topic}")

    # 1) START -> generate_query
    print("\n[STEP] generate_query")
    outputs = generate_query(state, config)
    for k, v in outputs.items():
        setattr(state, k, v)

    # 2) Loop to do web research and summarizing until 'finalize_summary'
    while True:
        # web_research
        print("\n[STEP] web_research")
        outputs = web_research(state, config)
        for k, v in outputs.items():
            setattr(state, k, v)

        # summarize_sources
        print("\n[STEP] summarize_sources")
        outputs = summarize_sources(state, config)
        for k, v in outputs.items():
            setattr(state, k, v)

        # reflect_on_summary
        print("\n[STEP] reflect_on_summary")
        outputs = reflect_on_summary(state, config)
        for k, v in outputs.items():
            setattr(state, k, v)

        # Decide next step
        print("[INFO] Checking if we should finalize or loop again...")
        next_node = route_research(state, config)
        if next_node == "finalize_summary":
            # finalize_summary
            print("\n[STEP] finalize_summary")
            outputs = finalize_summary(state)
            for k, v in outputs.items():
                setattr(state, k, v)
            break
        else:
            print("[INFO] Route says: do more web research.\n")

    return state

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <research_question>")
        sys.exit(1)

    research_question = " ".join(sys.argv[1:])

    # Run the pipeline
    final_state = run_research_pipeline(research_question)

    # Print final summary
    print("\n[FINAL SUMMARY]")
    print(final_state.running_summary or "[No summary produced]")

if __name__ == "__main__":
    main()
