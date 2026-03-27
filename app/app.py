from fastapi import FastAPI
from app.rag import RAGPipeline
from app.schemas import QueryRequest, QueryResponse, AnswerResponse

app = FastAPI(title="RAG API", version="0.1.0")

rag = RAGPipeline()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/query")
def query_get(q: str):
    results = rag.query(q)
    return {"query": q, "results": results}


@app.post("/query", response_model=QueryResponse)
def query_post(request: QueryRequest):
    results = rag.query(request.question, k=request.k)
    return {
        "query": request.question,
        "results": results,
    }

@app.post("/answer", response_model=AnswerResponse)
def answer(request: QueryRequest):
    result = rag.generate_answer(request.question, k=request.k)

    return {
        "query": request.question,
        "answer": result["answer"],
        "context": result["context"],
    }