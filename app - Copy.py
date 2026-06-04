import streamlit as st

from config import UPWORK_API_DOC_PATH
from rag import answer_question, retrieve_relevant_chunks


st.set_page_config(page_title="Upwork API Support Bot", layout="wide")

st.title("Upwork API Support Bot")
st.caption("RAG assistant grounded only in the provided Upwork API documentation.")

with st.sidebar:
    st.subheader("Document")
    st.write(str(UPWORK_API_DOC_PATH))
    st.subheader("Sample Questions")
    sample = st.radio(
        "Choose a question",
        [
            "What is the specific request-per-second rate limit for the Upwork API, and is it enforced per Key or per IP?",
            "How long is an OAuth access token valid for?",
            "Can I use a Client Credentials Grant to access a user's private contract details?",
        ],
        index=None,
    )

question = st.text_input(
    "Ask a developer question about the Upwork API",
    value=sample or "",
    placeholder="Example: How long is an OAuth access token valid for?",
)

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Retrieving sources and asking the model..."):
        try:
            result = answer_question(question.strip())
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    st.subheader("Answer")
    st.write(result.answer)

    st.metric("API latency", f"{result.latency_seconds:.2f} seconds")

    st.subheader("Sources")
    for idx, source in enumerate(result.sources, start=1):
        page = source.metadata.get("page")
        label = f"Source {idx}"
        if isinstance(page, int):
            label += f" - page {page + 1}"
        with st.expander(label, expanded=True):
            st.text(source.page_content)

with st.expander("Preview top 3 retrieved chunks without calling the LLM"):
    preview_question = st.text_input("Retrieval-only query", key="preview_query")
    if preview_question:
        for idx, source in enumerate(retrieve_relevant_chunks(preview_question), start=1):
            st.markdown(f"**Source {idx}**")
            st.text(source.page_content)
