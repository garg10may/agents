from agentic_core.agent import Agent
from agentic_core.memory import Memory

def run_debate_pipeline():
    agent_configs = [
        {"name": "Pro Agent", "system_prompt": "Argue in favor of AI safety regulations.", "toolset": ["summarize"]},
        {"name": "Con Agent", "system_prompt": "Argue against AI safety regulations.", "toolset": ["summarize"]},
        {"name": "Moderator", "system_prompt": "Moderate the debate and summarize the arguments.", "toolset": ["summarize"]}
    ]
    memory = Memory()
    agents = [Agent(**cfg) for cfg in agent_configs]
    memory.set('pipeline', [a.name for a in agents])
    steps = []
    current_agent = agents[0].name
    input_msg = "Debate: Should there be strict AI safety regulations?"
    for round in range(2):
        for agent in agents[:-1]:  # Pro and Con
            answer, agent_steps = agent.act(input_msg)
            steps.extend(agent_steps)
            input_msg = answer
    # Moderator summarizes
    moderator = agents[-1]
    answer, agent_steps = moderator.act(input_msg)
    steps.extend(agent_steps)
    print("\n--- Debate Reasoning Steps ---\n")
    for step in steps:
        print(step)
    print("\n--- Final Output ---\n")
    print(answer)

if __name__ == "__main__":
    run_debate_pipeline()
