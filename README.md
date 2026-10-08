# 🎥 YouTube RAG Assistant

A Retrieval-Augmented Generation (RAG) based chatbot that allows users to ask questions about YouTube videos.

The application fetches the YouTube video transcript, splits it into smaller chunks, converts the chunks into embeddings, stores them in a FAISS vector database, retrieves the most relevant information for a user's question, and uses a local Llama 3.2 model through Ollama to generate an answer.

## 🚀 Tech Stack

- Python
- Streamlit
- LangChain
- YouTube Transcript API
- HuggingFace Embeddings
- FAISS
- Ollama
- Llama 3.2
- Sentence Transformers

## 🔄 How It Works

YouTube Video
↓
Transcript
↓
Text Chunking
↓
HuggingFace Embeddings
↓
FAISS Vector Store
↓
Similarity Search
↓
Relevant Context
↓
Llama 3.2 via Ollama
↓
Final Answer

## ✨ Features

- 🎥 Enter a YouTube URL or Video ID
- 📄 Automatically fetch the video transcript
- ✂️ Split transcripts into manageable chunks
- 🧠 Generate multilingual embeddings
- 🔎 Retrieve relevant transcript sections using FAISS
- 🤖 Generate answers using Llama 3.2 locally
- 💬 Interactive Streamlit chat interface
- 💡 Quick question buttons
- 🔍 View retrieved transcript chunks
- 🟢 Local AI processing using Ollama

## 🤝 Learning & Development

This project was created as part of my learning journey into
RAG, LangChain, embeddings, vector databases, and local LLMs.

I used GPT as a learning and development assistant to understand
concepts, debug issues, improve the implementation, and build the
Streamlit interface.

The project helped me understand how a complete RAG pipeline works
from document retrieval to final LLM-generated responses.

## 🧠 What I Learned

- How Retrieval-Augmented Generation works
- How YouTube transcripts can be used as a knowledge source
- How text chunking works
- What embeddings are and why they are needed
- How FAISS performs similarity search
- How retrievers work in LangChain
- How prompts provide context to an LLM
- How to run Llama 3.2 locally using Ollama
- How to build an AI application using Streamlit
- How different components work together in a RAG pipeline

## ▶️ Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER