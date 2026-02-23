# RAG Vietnamese QA Chatbot

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-green)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Local RAG pipeline enabling semantic document search and question answering using **Ollama** LLMs, **ChromaDB** vector database, and **Streamlit** interface. 

This application allows users to upload PDF or image file and ask questions directly about the document content.

Built for learning and experimenting with:
- RAG pipelines
- Local LLM inference
- Vector search systems

## Demo

![Demo](images/demo.gif)

## Features
- Upload PDF / Image files
- Embedding with `qwen3-embedding:0.6b`
- Vector storage using ChromaDB (persistent)
- Using `"qwen3:1.7b", "deepseek-r1", "llama3.2:3b"` for LLM inferences
- Using `Streamlit` for Web-UI

## Tech Stack

- **Language**: Python 3.10+
- **Text splitting**: LangChain
- **Data processing**: pandas, kagglehub
- **Vector Database**: ChromaDB (persistent)
- **LLM & Embedding**: Ollama (local inference)

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/your-username/rag-document-reader.git
    
    cd rag-document-reader

cd rag-document-reader

    cd rag-vietnamese-qa
    ```

2.  Create and activate virtual environment:

    ```bash
    python -m venv .venv

    .venv\Scripts\activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Install Ollama & pull models:
    - Download Ollama: https://ollama.com/download
    - Pull embedding & LLM models:

      ```bash
      ollama pull qwen3-embedding:0.6b

      ollama pull qwen3:1.7b

      ollama pull deepseek-r1

      ollama pull llama3.2:3b

      ollama serve
      ```

5.  Run the chatbot with Streamlit:
    ```bash
    streamlit run app.py
    ```

