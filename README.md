# Production-Style RAG System with LangChain
 
> A production-grade Retrieval-Augmented Generation (RAG) system that delivers context-aware answers over domain documents — with hybrid retrieval, re-ranking, prompt versioning, offline evaluation, and latency instrumentation.
 
🚀 **Live Demo:** [kirthika-rag-lanchain-demo.streamlit.app](https://kirthika-rag-lanchain-demo.streamlit.app)
 
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green)](https://python.langchain.com/)
[![FAISS](https://img.shields.io/badge/VectorStore-FAISS-orange)](https://faiss.ai/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B)](https://streamlit.io/)
 
---
 
## Overview
 
This project demonstrates a production-ready RAG pipeline built with LangChain, FAISS, and FastAPI. It goes beyond a simple Q&A chatbot by implementing the architectural decisions that matter in real enterprise deployments: retrieval quality, cost control, evaluation, and iterative prompt improvement.
 
The system indexes domain documents, retrieves the most relevant context using hybrid search + re-ranking, and generates grounded, accurate answers via an LLM — all with response caching and latency tracking built in.

This project implements an end-to-end RAG system that:

- ingests local documents

- generates embeddings

- performs hybrid retrieval

- re-ranks retrieved contexts

- generates grounded answers using an LLM
 
---

## Result:
https://github.com/user-attachments/assets/6922b737-056c-4b9c-9229-1283604e9a3d


---
 
## Author
 
**Kirthika Srinivasan** — Applied AI Engineer | Melbourne, VIC  
[LinkedIn](https://www.linkedin.com/in/kirthikasrinivasan) · [GitHub](https://github.com/Kirthika-Srinivasan) · [Live Demo](https://kirthika-rag-lanchain-demo.streamlit.app)
