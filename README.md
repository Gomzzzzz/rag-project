# RAG API Project

A simple Retrieval-Augmented Generation (RAG) API built with FastAPI, FAISS, Hugging Face embeddings, and Docker.

This project ingests a PDF document, splits it into chunks, creates vector embeddings, stores them in a FAISS index, and exposes API endpoints for:
- retrieving relevant chunks from the document
- generating an answer based on retrieved context

---

## Features

- PDF ingestion
- Text chunking with overlap
- Semantic retrieval using FAISS
- FastAPI-based API service
- Persistent FAISS index loading
- Answer generation (basic)
- Dockerized app
- Swagger docs via `/docs`

---

## Project Structure
rag-project/
├── app/
│ ├── app.py
│ ├── rag.py
│ └── schemas.py
├── data/
│ └── sample.pdf
├── storage/
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md


---

## Tech Stack

- Python
- FastAPI
- Uvicorn
- LangChain
- Hugging Face Embeddings (`all-MiniLM-L6-v2`)
- FAISS
- Transformers (`distilgpt2` for generation)
- Docker

---

## Architecture

