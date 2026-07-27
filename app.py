import os
import tempfile
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai

load_dotenv()

st.set_page_config(
    page_title="NEXUS-RAG | Autonomous AI Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

api_key = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "doc_chunks" not in st.session_state: st.session_state.doc_chunks = []
if "doc_embeddings" not in st.session_state: st.session_state.doc_embeddings = []
if "doc_stats" not in st.session_state: st.session_state.doc_stats = {"files": 0, "chunks": 0}
if "latest_sources" not in st.session_state: st.session_state.latest_sources = []

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

if process_btn:
    if not api_key:
        st.sidebar.error("⚠️ API Key not found in environment or secrets.")
    elif not uploaded_files:
        st.sidebar.error("⚠️ Upload at least one PDF file.")
    elif not client:
        st.sidebar.error("⚠️ Google GenAI Client not initialized.")
    else:
        with st.spinner("⚡ Generating Semantic Embeddings (text-embedding-001)..."):
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
            
            embeddings = []
            for chunk in chunks:
                response = client.models.embed_content(
                    model="text-embedding-001",
                    contents=chunk.page_content
                )
                embeddings.append(response.embedding.values)
            
            st.session_state.doc_chunks = chunks
            st.session_state.doc_embeddings = np.array(embeddings)
            st.session_state.doc_stats = {"files": len(uploaded_files), "chunks": len(chunks)}
            st.sidebar.success("✨ Semantic Vector Index Online!")

def semantic_search(query, k=4):
    if len(st.session_state.doc_chunks) == 0 or client is None:
        return []
    
    q_response = client.models.embed_content(
        model="text-embedding-001",
        contents=query
    )
    q_vec = np.array(q_response.embedding.values)
    
    doc_matrix = st.session_state.doc_embeddings
    similarities = np.dot(doc_matrix, q_vec) / (np.linalg.norm(doc_matrix, axis=1) * np.linalg.norm(q_vec) + 1e-10)
    
    top_indices = np.argsort(similarities)[::-1][:k]
    return [st.session_state.doc_chunks[i] for i in top_indices]

st.markdown("""
<div class="hero-container">
    <div class="hero-title">⚡ NEXUS-RAG Intelligence</div>
    <div class="hero-subtitle">Autonomous Semantic Q&A powered by Google Embeddings & Gemini Flash</div>
</div>
""", unsafe_allow_html=True)

chat_col, source_col = st.columns([7, 3], gap="medium")

with chat_col:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask a question across your documents...")
    if user_query:
        if len(st.session_state.doc_chunks) == 0:
            st.warning("⚠️ Please upload and initialize documents in the sidebar first.")
        elif not client:
            st.error("⚠️ Google API Client not initialized.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_query)
                
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                full_response = ""
                try:
                    retrieved_docs = semantic_search(user_query)
                    st.session_state.latest_sources = retrieved_docs
                    
                    context_text = "\n\n".join([
                        f"Content: {d.page_content}\n[Source: {d.metadata.get('source')}, Page: {d.metadata.get('page', 0)+1}]" 
                        for d in retrieved_docs
                    ])
                    
                    prompt = f"Answer accurately using ONLY the provided context. If the answer is absent, state that you cannot find it. Cite sources clearly.\n\nContext:\n{context_text}\n\nQuestion: {user_query}"
                    
                    response = client.models.generate_content(
                        model='gemini-3.6-flash',
                        contents=prompt
                    )
                    
                    full_response = response.text
                    
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
        st.markdown(f"<p style='color: #94A3B8; font-size: 0.85rem;'>Retrieved {len(st.session_state.latest_sources)} semantic chunks</p>", unsafe_allow_html=True)
        for doc in st.session_state.latest_sources:
            src_name = doc.metadata.get('source', 'Unknown')
            page_num = doc.metadata.get('page', 0) + 1
            preview_text = doc.page_content[:220].replace('\n', ' ')
            st.markdown(f'<div class="glass-card" style="font-size: 0.85rem;"><div style="font-weight: 600; color: #06B6D4; margin-bottom: 4px;">📄 {src_name} (Page {page_num})</div><div style="color: #94A3B8; font-style: italic;">"{preview_text}..."</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="glass-card" style="text-align: center; color: #94A3B8; padding: 2rem 1rem;"><div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📡</div>Ask a query to inspect live semantic retrieval weights and citations.</div>', unsafe_allow_html=True)