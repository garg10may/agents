from agentic_core.agent import Agent
from agentic_core.memory import Memory
from agentic_core.routing import default_routing_fn

def run_blog_pipeline():
    agent_configs = [
        {"name": "Researcher", "system_prompt": "You are a research agent. Use web search, Wikipedia, and summarization to gather and condense information for the topic provided.", "toolset": ["web_search", "wikipedia_search", "summarize"]},
        {"name": "Writer", "system_prompt": "You are a blog writer. Use summarization, translation, and entity extraction to write a clear, engaging article from the research notes provided.", "toolset": ["summarize", "translate", "extract_entities"]},
        {"name": "Reviewer", "system_prompt": "You are a critical reviewer. Use sentiment analysis and summarization to review and improve the article provided.", "toolset": ["summarize", "sentiment_analysis"]}
    ]
    memory = Memory()
    agents = [Agent(**cfg) for cfg in agent_configs]
    memory.set('pipeline', [a.name for a in agents])
    steps = []
    current_agent = agents[0].name
    input_msg = "Write a blog post about the latest AI breakthrough, then review it for sentiment and summarize the review."
    while True:
        agent = next(a for a in agents if a.name == current_agent)
        answer, agent_steps = agent.act(input_msg)
        steps.extend(agent_steps)
        idx = [a.name for a in agents].index(current_agent)
        if idx + 1 >= len(agents):
            break
        current_agent = agents[idx + 1].name
        input_msg = answer
    print("\n--- Reasoning Steps ---\n")
    for step in steps:
        print(step)
    print("\n--- Final Output ---\n")
    print(answer)

if __name__ == "__main__":
    run_blog_pipeline()
