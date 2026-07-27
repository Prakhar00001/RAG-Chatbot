import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="NEXUS-RAG | Autonomous AI Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .stApp { background-color: #090A0F; color: #F8FAFC; font-family: 'Inter', sans-serif; }
    .hero-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(6, 182, 212, 0.05) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 16px; padding: 2rem; text-align: center; margin-bottom: 2rem;
    }
    .hero-title { font-size: 2.5rem; font-weight: 800; background: linear-gradient(90deg, #6366F1, #06B6D4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero-subtitle { color: #94A3B8; font-size: 1.05rem; }
    .glass-card { background: rgba(18, 20, 28, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem; }
    .metric-box { background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 8px; padding: 0.75rem; text-align: center; }
</style>
""", unsafe_allow_html=True)

if not os.environ.get("GOOGLE_API_KEY"):
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass

# Session State
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "retriever" not in st.session_state: st.session_state.retriever = None
if "doc_stats" not in st.session_state: st.session_state.doc_stats = {"files": 0, "chunks": 0}
if "latest_sources" not in st.session_state: st.session_state.latest_sources = []

# Sidebar
with st.sidebar:
    st.markdown("### 📁 Knowledge Base Ingestion")
    uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
    process_btn = st.button("🚀 Initialize Pipeline")
    
    st.markdown("---")
    st.markdown("### 📊 System Telemetry")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f'<div class="metric-box"><div style="font-size:0.75rem;color:#94A3B8;">DOCS</div><div style="font-size:1.25rem;font-weight:700;color:#06B6D4;">{st.session_state.doc_stats["files"]}</div></div>', unsafe_allow_html=True)
    with col_s2:
        st.markdown(f'<div class="metric-box"><div style="font-size:0.75rem;color:#94A3B8;">CHUNKS</div><div style="font-size:1.25rem;font-weight:700;color:#6366F1;">{st.session_state.doc_stats["chunks"]}</div></div>', unsafe_allow_html=True)

    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_history = []
        st.session_state.latest_sources = []
        st.rerun()

# Pipeline Processing (Pure Python BM25 Indexing)
if process_btn:
    if not os.environ.get("GOOGLE_API_KEY"):
        st.sidebar.error("⚠️ API Key not found in environment or secrets.")
    elif not uploaded_files:
        st.sidebar.error("⚠️ Upload at least one PDF file.")
    else:
        with st.spinner("⚡ Initializing Pure-Python Keyword & Semantic Index..."):
            all_docs = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                try:
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = uploaded_file.name
                    all_docs.extend(docs)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(all_docs)
            
            # Pure Python BM25 Retriever - Zero C++ DLL dependencies
            bm25_retriever = BM25Retriever.from_documents(chunks)
            bm25_retriever.k = 4
            st.session_state.retriever = bm25_retriever
            
            st.session_state.doc_stats = {"files": len(uploaded_files), "chunks": len(chunks)}
            st.sidebar.success("✨ Index Online!")

# Main Workspace
st.markdown("""
<div class="hero-container">
    <div class="hero-title">⚡ NEXUS-RAG Intelligence</div>
    <div class="hero-subtitle">Autonomous multi-document Q&A powered by Gemini 1.5 Flash</div>
</div>
""", unsafe_allow_html=True)

chat_col, source_col = st.columns([7, 3], gap="medium")

with chat_col:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask a question across your documents...")
    if user_query:
        if st.session_state.retriever is None:
            st.warning("⚠️ Please upload and process documents in the sidebar first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_query)
                
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                full_response = ""
                try:
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, streaming=True)
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", "Answer accurately using ONLY the provided context. If the answer is absent, state that you cannot find it. Cite sources clearly.\n\nContext:\n{context}"),
                        ("human", "{question}")
                    ])
                    
                    retrieved_docs = st.session_state.retriever.invoke(user_query)
                    st.session_state.latest_sources = retrieved_docs
                    
                    context_text = "\n\n".join([
                        f"Content: {d.page_content}\n[Source: {d.metadata.get('source')}, Page: {d.metadata.get('page', 0)+1}]" 
                        for d in retrieved_docs
                    ])
                    
                    chain = (
                        {"context": lambda x: context_text, "question": RunnablePassthrough()}
                        | prompt_template
                        | llm
                        | StrOutputParser()
                    )
                    
                    for chunk in chain.stream(user_query):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    
                    sources = sorted(set(f"{d.metadata.get('source')} (Page {d.metadata.get('page', 0)+1})" for d in retrieved_docs))
                    if sources:
                        full_response += f"\n\n**Sources:** {', '.join(sources)}"
                    
                    message_placeholder.markdown(full_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    message_placeholder.error(f"Error: {str(e)}")

with source_col:
    st.markdown("### 🔍 Live Source Inspector")
    if st.session_state.latest_sources:
        st.markdown(f"<p style='color: #94A3B8; font-size: 0.85rem;'>Retrieved {len(st.session_state.latest_sources)} chunks</p>", unsafe_allow_html=True)
        for doc in st.session_state.latest_sources:
            src_name = doc.metadata.get('source', 'Unknown')
            page_num = doc.metadata.get('page', 0) + 1
            preview_text = doc.page_content[:220].replace('\n', ' ')
            st.markdown(f'<div class="glass-card" style="font-size: 0.85rem;"><div style="font-weight: 600; color: #06B6D4; margin-bottom: 4px;">📄 {src_name} (Page {page_num})</div><div style="color: #94A3B8; font-style: italic;">"{preview_text}..."</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="glass-card" style="text-align: center; color: #94A3B8; padding: 2rem 1rem;"><div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📡</div>Ask a query to inspect live context retrieval weights and citations.</div>', unsafe_allow_html=True)