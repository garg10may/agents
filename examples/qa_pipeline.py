from agentic_core.agent import Agent
from agentic_core.memory import Memory

def run_qa_pipeline():
    agent_configs = [
        {"name": "QA Researcher", "system_prompt": "You are a QA agent. Use web search and summarization to answer questions.", "toolset": ["web_search", "summarize"]},
        {"name": "QA Reviewer", "system_prompt": "You are a reviewer. Use sentiment analysis to review the answer.", "toolset": ["sentiment_analysis"]}
    ]
    memory = Memory()
    agents = [Agent(**cfg) for cfg in agent_configs]
    memory.set('pipeline', [a.name for a in agents])
    steps = []
    current_agent = agents[0].name
    input_msg = "What is the latest in AI research?"
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
    run_qa_pipeline()
