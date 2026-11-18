"""Working baseline chatbot with optional RAG context."""
import streamlit as st
from llm import ask_llm
from rag import build_context, retrieve

st.title("Project VITAL — LLM + RAG Lab")
enable_rag = st.sidebar.checkbox("Enable RAG", value=False)

if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask a question")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # TODO 4: Add an Enable RAG toggle; retrieve documents and pass context.
    context = None
    source_note = None
    if enable_rag:
        hits = retrieve(question, k=3)
        context = build_context(hits)
        if hits:
            names = ", ".join(name for name, _, _ in hits)
            source_note = f"Sources: {names}"
        else:
            source_note = "No documents retrieved"

    try:
        answer = ask_llm(question, context) if enable_rag else ask_llm(question)
    except Exception as exc:
        answer = f"Ollama unavailable: {exc}"

    with st.chat_message("assistant"):
        st.write(answer)
        if source_note:
            st.caption(source_note)
    st.session_state.messages.append({"role": "assistant", "content": answer})
