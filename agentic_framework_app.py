import streamlit as st
from agentic_framework import agentic_function_calling_agent

st.set_page_config(page_title="Agentic Function Calling Agent", layout="centered")
st.title("🛠️ Universal Tool-Using Agent")

with st.form("framework_form"):
    user_goal = st.text_area("Enter your complex goal:")
    verbose = st.checkbox("Show reasoning steps", value=True)
    submitted = st.form_submit_button("Run Agent")

if submitted and user_goal:
    with st.spinner("Agent reasoning, using tools, and iterating..."):
        try:
            answer, steps = agentic_function_calling_agent(user_goal, verbose=verbose)
            st.success("Agent completed the task!")
            if verbose and steps:
                st.markdown("### Reasoning Steps:")
                for step in steps:
                    st.markdown(step)
            st.markdown("### Final Agent Answer:")
            st.markdown(answer)
        except Exception as e:
            st.error(f"Error: {e}")
