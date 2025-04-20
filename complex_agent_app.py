import streamlit as st
from complex_agent import agentic_reasoning_loop

st.set_page_config(page_title="Complex Agentic Workflow", layout="centered")
st.title("🤖 Complex Tool-Using Agent")

with st.form("agent_form"):
    user_goal = st.text_area("Enter your complex goal:")
    submitted = st.form_submit_button("Run Agent")

if submitted and user_goal:
    with st.spinner("Agent reasoning, using tools, and iterating..."):
        try:
            result = agentic_reasoning_loop(user_goal)
            st.success("Agent completed the task!")
            st.markdown("### Final Agent Answer:")
            st.markdown(result)
        except Exception as e:
            st.error(f"Error: {e}")
