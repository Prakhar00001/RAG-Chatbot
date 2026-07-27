# ⚡ NEXUS-RAG | Autonomous AI Intelligence

> An enterprise-grade, zero-DLL, pure-Python Retrieval-Augmented Generation (RAG) system built with **Streamlit**, **BM25 text retrieval**, and Google's official **Gemini GenAI SDK (`gemini-3.6-flash`)**.

---

## 🚀 Overview

**NEXUS-RAG** is an autonomous multi-document intelligence platform designed to ingest complex technical documents (PDFs), index them efficiently using high-performance keyword matching, and answer queries with precise context grounding and strict source citations.

Engineered specifically to bypass native Windows C++ compilation and DLL initialization errors (`WinError 1114`), NEXUS-RAG relies entirely on a robust, pure-Python architecture that boots up instantly and runs smoothly across any environment.

---

## 💎 Key Features

* **Zero C++ Binaries / DLL Error Free:** Completely independent of heavy local ML runtime packages like PyTorch or FAISS, making it 100% stable on Windows, macOS, and Linux.
* **BM25 Precision Retriever:** Uses optimized keyword-based document indexing to map user queries directly to exact paragraphs, headings, and tables.
* **Powered by Google Gemini 3.6 Flash:** Integrates the official, high-speed `google-genai` SDK for ultra-low latency generation.
* **Modern Glassmorphic UI:** Features a dark-mode cyberpunk/glassmorphism interface built with Streamlit custom HTML/CSS.
* **Inline Chat Form Layout:** Designed with a chat flow where the input form sits directly below the latest response (similar to ChatGPT and Claude).
* **Live Source Inspector:** Real-time side panel that displays retrieved document chunks, source filenames, and precise page numbers for complete auditability.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **UI Framework:** Streamlit
* **Document Parsing:** `pypdf`
* **Text Splitting & Retrieval:** `langchain-text-splitters`, `langchain-community` (`BM25Retriever`)
* **AI Engine:** Google GenAI SDK (`gemini-3.6-flash`)
* **Configuration:** `python-dotenv`, Streamlit Secrets

---

## 📦 Project Structure

```text
RAG-Chatbot/
│
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .env                 # Local environment variables (API keys)
└── README.md            # Project documentation



⚙️ Installation & Local Setup

1. Clone the Repository

git clone [https://github.com/Prakhar00001/RAG-Chatbot.git]

cd RAG-Chatbot

2. Create and Activate Virtual Environment

python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\Activate
# On macOS/Linux:
source venv/bin/activate


3. Install Dependencies

pip install -r requirements.txt


4. Configure Environment Variables


Create a .env file in the root directory and add your Google Gemini API key:


GOOGLE_API_KEY="your_google_gemini_api_key_here"


5. Run the Application

streamlit run app.py


💡 How to Use


Upload Documents: Use the sidebar file uploader to upload one or more PDF files (e.g., lecture notes, technical whitepapers, manuals).

Initialize Pipeline: Click "🚀 Initialize Pipeline" to parse pages, slice them into optimized text chunks, and build the BM25 index.

Ask Questions: Type your query into the inline chat bar below the message history.

Inspect Sources: Review the AI-generated answer alongside the Live Source Inspector panel to verify exact source documents and page numbers.


