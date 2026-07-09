import os
import tempfile
from typing import List

import fitz
import streamlit as st

from rag_chatbot import RAGPipeline

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="wide")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(embedding_model="local")
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_data(show_spinner=False)
def extract_text_from_pdf(uploaded_file) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        tf.write(uploaded_file.getvalue())
        temp_path = tf.name

    doc = fitz.open(temp_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    os.unlink(temp_path)
    return text


st.title("Autonomous PDF RAG Chatbot")
st.caption("Upload PDFs, ask questions, and get source-attributed answers.")
st.info("Step 1: upload one or more PDF files. Step 2: click Process PDFs. Step 3: ask a question in the chat box.")

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Process PDFs") and uploaded_files:
        documents: List[str] = []
        sources: List[str] = []
        progress = st.progress(0)
        for index, uploaded in enumerate(uploaded_files):
            text = extract_text_from_pdf(uploaded)
            documents.append(text)
            sources.append(uploaded.name)
            progress.progress((index + 1) / len(uploaded_files))
        st.session_state.pipeline.index_documents(documents, sources=sources)
        chunk_count = len(st.session_state.pipeline.chunks)
        source_names = ", ".join(sources)
        st.success(f"Processed {len(uploaded_files)} PDF file(s). Indexed {chunk_count} text chunk(s) from: {source_names}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your uploaded documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the indexed document chunks and forming an answer..."):
            answer = st.session_state.pipeline.answer_question(prompt)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
