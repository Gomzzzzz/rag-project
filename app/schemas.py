from pydantic import BaseModel, Field
from typing import List, Dict, Any


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    k: int = Field(default=5, ge=1, le=10, description="Number of chunks to retrieve")


class RetrievalResult(BaseModel):
    content: str
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    query: str
    results: List[RetrievalResult]
