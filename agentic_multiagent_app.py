import streamlit as st
from agentic_multiagent_framework import multiagent_pipeline

st.set_page_config(page_title="Multi-Agentic Workflow", layout="wide")
st.title("🤝 Modular Multi-Agentic Workflow")

with st.expander("ℹ️ Example: Research, Write, Review a Blog Post (AI News)", expanded=True):
    st.markdown("""
    **Pipeline:** Researcher → Writer → Reviewer  
    **Goal Example:**
    > Write a blog post about the latest AI breakthrough, then review it for sentiment and summarize the review.
    """)

# Default agent configs for the UI
DEFAULT_AGENT_CONFIGS = [
    {
        "name": "Researcher",
        "system_prompt": "You are a research agent. Use web search, Wikipedia, and summarization to gather and condense information for the topic provided.",
        "toolset": ["web_search", "wikipedia_search", "summarize"]
    },
    {
        "name": "Writer",
        "system_prompt": "You are a blog writer. Use summarization, translation, and entity extraction to write a clear, engaging article from the research notes provided.",
        "toolset": ["summarize", "translate", "extract_entities"]
    },
    {
        "name": "Reviewer",
        "system_prompt": "You are a critical reviewer. Use sentiment analysis and summarization to review and improve the article provided.",
        "toolset": ["summarize", "sentiment_analysis"]
    }
]

st.sidebar.header("🧩 Agent Pipeline Configuration")
num_agents = st.sidebar.number_input("Number of Agents", min_value=1, max_value=6, value=3)

agent_configs = []
for i in range(num_agents):
    with st.sidebar.expander(f"Agent {i+1} Settings", expanded=True):
        name = st.text_input(f"Name {i+1}", value=DEFAULT_AGENT_CONFIGS[i]["name"] if i < len(DEFAULT_AGENT_CONFIGS) else f"Agent {i+1}")
        system_prompt = st.text_area(f"System Prompt {i+1}", value=DEFAULT_AGENT_CONFIGS[i]["system_prompt"] if i < len(DEFAULT_AGENT_CONFIGS) else "Describe this agent's role...")
        toolset = st.multiselect(
            f"Tools {i+1}",
            ["web_search", "wikipedia_search", "summarize", "translate", "extract_entities", "sentiment_analysis", "calculator", "get_time", "get_date", "code_executor", "url_reader"],
            default=DEFAULT_AGENT_CONFIGS[i]["toolset"] if i < len(DEFAULT_AGENT_CONFIGS) else []
        )
        agent_configs.append({"name": name, "system_prompt": system_prompt, "toolset": toolset})

st.markdown("## 📝 Run a Multi-Agent Pipeline")
user_goal = st.text_area("Enter your multi-agent goal:", value="Write a blog post about the latest AI breakthrough, then review it for sentiment and summarize the review.")

if st.button("Run Multi-Agent Pipeline"):
    with st.spinner("Agents collaborating and reasoning..."):
        try:
            final_answer, steps = multiagent_pipeline(user_goal, agent_configs)
            st.success("Pipeline completed!")
            st.markdown("### Reasoning Steps:")
            for step in steps:
                st.markdown(step)
            st.markdown("### Final Output:")
            st.markdown(final_answer)
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.markdown("""
### 🚀 Coming Soon: Drag-and-Drop Visual Pipeline
- Visualize and reorder agents with connectors
- Add/remove agents with a modern UI (e.g., [React Flow](https://reactflow.dev/) or [Streamlit Components](https://docs.streamlit.io/library/components))
- For now, use the sidebar to configure your pipeline!
""")
