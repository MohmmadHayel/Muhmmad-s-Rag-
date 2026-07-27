import os
import streamlit as st
from RAG import VectorDBBuilder, RAGQueryEngine, llm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def get_rag_engine():
    faiss_path = os.path.join(SCRIPT_DIR, "faiss_index")
    builder = VectorDBBuilder(file_path=os.path.join(SCRIPT_DIR, "aboutme.txt"))

    if os.path.exists(faiss_path):
        builder.load(faiss_path)
    else:
        builder.build()
        builder.save(faiss_path)

    return RAGQueryEngine(builder.vectorDB, llm, k=4)


st.set_page_config(page_title="Ask About Mohammad", page_icon="🤖")
st.title("🤖 Ask About Mohammad Al-Hmoud")

rag = get_rag_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("اسأل عن مؤهلات محمد مشاريعه، أو مساره المهني...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("بفكر..."):
            response = rag.answer(query)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})