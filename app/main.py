from fastapi import FastAPI, HTTPException
from app.models import QueryRequest, QueryResponse
from app.graph import app_graph
from fastapi.responses import RedirectResponse

app = FastAPI(title="Agentic AI RAG API")

@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse(url="/docs")

@app.post("/query", response_model=QueryResponse)
def query_chatbot(request: QueryRequest):
    try:
        # Run LangGraph
        result = app_graph.invoke({"query": request.query, "context": [], "answer": "", "confidence_score": 0.0})
        
        return QueryResponse(
            query=request.query,
            final_answer=result.get("answer", ""),
            retrieved_context_chunks=result.get("context", []),
            confidence_score=result.get("confidence_score", 0.0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
