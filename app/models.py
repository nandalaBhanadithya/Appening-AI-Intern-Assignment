from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: List[str]
    confidence_score: float
