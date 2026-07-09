# Autonomous PDF RAG Chatbot

A production-style Retrieval-Augmented Generation chatbot for PDF documents with chunking, FAISS indexing, hybrid retrieval, citations, and a Streamlit UI.

## Features
- Multi-PDF upload and ingestion
- Recursive document chunking
- FAISS vector store and BM25 keyword retrieval
- Source-attributed answers with citations
- Session-aware conversation history
- Clear status output during indexing and answering

## Project structure
- [app.py](app.py) — Streamlit web app
- [rag_chatbot.py](rag_chatbot.py) — indexing and retrieval logic
- [tests/test_pipeline.py](tests/test_pipeline.py) — regression tests
- [Dockerfile](Dockerfile) — container deployment

## Setup
1. Open the project folder in terminal.
2. Create and activate a virtual environment:
   - `python -m venv .venv`
   - `.venv\Scripts\Activate.ps1`
3. Install dependencies:
   - `python -m pip install -r requirements.txt`
4. Copy [.env.example](.env.example) to .env and add your OpenAI credentials if you want OpenAI-backed responses.
5. Run the app:
   - `python -m streamlit run app.py --server.headless true --server.port 8501`
6. Open the printed local URL in your browser.

## Expected output
When you run the app, you should see:
- A Streamlit page titled "Autonomous PDF RAG Chatbot"
- An instruction box explaining the 3-step workflow
- A sidebar for PDF upload
- A success message after processing files such as: "Processed 2 PDF file(s). Indexed 12 text chunk(s) from: file1.pdf, file2.pdf"
- A chat response that includes the answer text and the source names

## Deploy
- Streamlit Community Cloud: deploy [app.py](app.py) directly
- Docker:
  - `docker build -t rag-chatbot .`
  - `docker run -p 8501:8501 rag-chatbot`
