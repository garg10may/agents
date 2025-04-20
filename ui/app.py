import streamlit as st
from agentic_core.agent import Agent
from agentic_core.memory import Memory
from agentic_core.tool import ToolRegistry
from agentic_core.routing import default_routing_fn
from agentic_core.critic import Critic
from agentic_core.planner import Planner

st.set_page_config(page_title="Advanced Multi-Agentic Workflow", layout="wide")
st.title("🤖 Advanced Modular Multi-Agentic Framework")

with st.expander("ℹ️ Example: Research, Write, Review a Blog Post", expanded=True):
    st.markdown("""
    **Pipeline:** Researcher → Writer → Reviewer  
    **Goal Example:**
    > Write a blog post about the latest AI breakthrough, then review it for sentiment and summarize the review.
    """)

# Sidebar config
st.sidebar.header("🧩 Agent Pipeline Configuration")
num_agents = st.sidebar.number_input("Number of Agents", min_value=1, max_value=6, value=3)

agent_configs = []
for i in range(num_agents):
    with st.sidebar.expander(f"Agent {i+1} Settings", expanded=True):
        name = st.text_input(f"Name {i+1}", value=f"Agent {i+1}")
        system_prompt = st.text_area(f"System Prompt {i+1}", value="Describe this agent's role...")
        toolset = st.multiselect(
            f"Tools {i+1}",
            ["web_search", "wikipedia_search", "summarize", "translate", "extract_entities", "sentiment_analysis", "calculator", "get_time", "get_date", "code_executor", "url_reader"],
            default=[]
        )
        agent_configs.append({"name": name, "system_prompt": system_prompt, "toolset": toolset})

st.markdown("## 📝 Run a Multi-Agent Pipeline")
user_goal = st.text_area("Enter your multi-agent goal:", value="Write a blog post about the latest AI breakthrough, then review it for sentiment and summarize the review.")

if st.button("Run Multi-Agent Pipeline"):
    with st.spinner("Agents collaborating and reasoning..."):
        # Setup agents & memory
        memory = Memory()
        agents = [Agent(**cfg) for cfg in agent_configs]
        # Setup pipeline in memory for routing
        memory.set('pipeline', [a.name for a in agents])
        steps = []
        current_agent = agents[0].name
        input_msg = user_goal
        while True:
            agent = next(a for a in agents if a.name == current_agent)
            answer, agent_steps = agent.act(input_msg)
            steps.extend(agent_steps)
            idx = [a.name for a in agents].index(current_agent)
            if idx + 1 >= len(agents):
                break
            current_agent = agents[idx + 1].name
            input_msg = answer
        st.success("Pipeline completed!")
        st.markdown("### Reasoning Steps:")
        for step in steps:
            st.markdown(step)
        st.markdown("### Final Output:")
        st.markdown(answer)

st.markdown("---")
st.markdown("""
### 🚀 Coming Soon: Drag-and-Drop Visual Pipeline
- Visualize and reorder agents with connectors
- Add/remove agents with a modern UI (e.g., React Flow or Streamlit Components)
- For now, use the sidebar to configure your pipeline!
""")
