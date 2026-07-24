import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(
    page_title="Autonomous PDF RAG Chatbot (Gemini)",
    page_icon="🤖",
    layout="wide"
)

# Sidebar Configuration for API Key & File Uploads
with st.sidebar:
    st.title("⚙️ Configuration")
    api_key_input = st.text_input("Gemini API Key", type="password")
    
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input
    elif "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        
    st.markdown("---")
    st.subheader("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    process_btn = st.button("🚀 Process Documents", type="primary")

st.title("🤖 Autonomous PDF RAG Chatbot (Gemini Powered)")
st.markdown("State-of-the-art hybrid RAG pipeline combining dense vector search (FAISS), keyword lookup (BM25), and Google Gemini streaming generation.")

# Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Document Processing Pipeline
if process_btn:
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("⚠️ Please provide a valid Google Gemini API key.")
    elif not uploaded_files:
        st.error("⚠️ Please upload at least one PDF file.")
    else:
        with st.spinner("Processing documents (Chunking, Gemini Embedding, Hybrid Indexing)..."):
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
            
            # Recursive Chunking
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(all_docs)
            
            # Gemini Dense Embeddings & FAISS Indexing
            embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
            faiss_vectorstore = FAISS.from_documents(chunks, embeddings)
            faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 4})
            
            # BM25 Sparse Keyword Retriever
            bm25_retriever = BM25Retriever.from_documents(chunks)
            bm25_retriever.k = 4
            
            # Hybrid Ensemble Retriever (60% Dense, 40% Sparse)
            ensemble_retriever = EnsembleRetriever(
                retrievers=[faiss_retriever, bm25_retriever],
                weights=[0.6, 0.4]
            )
            st.session_state.retriever = ensemble_retriever
            st.success(f"Successfully processed {len(uploaded_files)} files into {len(chunks)} chunks. Hybrid index online!")

# Render Conversation History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input & LLM Generation Execution
user_query = st.chat_input("Ask a question about your documents...")
if user_query:
    if st.session_state.retriever is None:
        st.warning("⚠️ Please upload and process documents in the sidebar first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, streaming=True)
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", "Answer the question accurately using ONLY the provided context. If the answer is not contained within the context, state that you cannot find the information. Cite sources clearly.\n\nContext:\n{context}"),
                    ("human", "{question}")
                ])
                
                retrieved_docs = st.session_state.retriever.invoke(user_query)
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
                
                # Append Clean Citations Footer
                sources = sorted(set(f"{d.metadata.get('source')} (Page {d.metadata.get('page', 0)+1})" for d in retrieved_docs))
                if sources:
                    full_response += f"\n\n**Sources:** {', '.join(sources)}"
                
                message_placeholder.markdown(full_response)
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
            except Exception as e:
                message_placeholder.error(f"Error generating response: {str(e)}")