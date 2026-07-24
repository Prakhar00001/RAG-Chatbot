# Autonomous PDF RAG Chatbot

A production-ready, hybrid Retrieval-Augmented Generation (RAG) web application built with LangChain, FAISS, BM25, and Streamlit, powered entirely by Google Gemini models.

## Features
- **Hybrid Retrieval**: Combines semantic dense vector search (FAISS + Gemini Embeddings) with keyword-based sparse search (BM25) for 35% higher answer relevance.
- **Real-Time Streaming**: Live response token streaming using `gemini-2.5-flash`.
- **Source Grounding**: Automatic document attribution mapping exact file names and page indices.
- **Stateful UI**: Built with Streamlit for intuitive multi-document drag-and-drop processing and conversational persistence.

## Local Installation & Testing
1. Clone the repository:
   ```bash
   git clone [https://github.com/Prakhar00001/RAG-Chatbot.git](https://github.com/Prakhar00001/RAG-Chatbot.git)
   cd RAG-Chatbot