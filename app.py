import streamlit as st
from agentic_flow import agentic_blog_flow

st.set_page_config(page_title="Agentic Blog Generator", layout="centered")
st.title("📝 Agentic Blog Post Generator")

with st.form("blog_form"):
    topic = st.text_input("Enter a blog topic:")
    submitted = st.form_submit_button("Generate Blog Post")

if submitted and topic:
    with st.spinner("Generating blog post. This may take a few moments..."):
        try:
            result = agentic_blog_flow(topic)
            st.success("Blog post generated!")
            st.markdown(result)
        except Exception as e:
            st.error(f"Error: {e}")
