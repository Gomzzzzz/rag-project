from fastapi import FastAPI
from app.rag import RAGPipeline

app = FastAPI()

rag = RAGPipeline()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/query")
def query(q: str):
    results = rag.query(q)
    return {"query": q, "results": results}