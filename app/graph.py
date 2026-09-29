from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.config import GOOGLE_API_KEY, PINECONE_INDEX_NAME
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Define State
class GraphState(TypedDict):
    query: str
    context: List[str]
    answer: str
    confidence_score: float

# Initialize embeddings and language model
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=GOOGLE_API_KEY)
    vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
except Exception as e:
    # Allows the app to start even if keys are not set, failing only on invocation
    retriever = None

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0, google_api_key=GOOGLE_API_KEY)

class AnswerFormat(BaseModel):
    answer: str = Field(description="The synthesized answer based on the context. If the context does not contain the answer, state that the information is not available in the eBook context.")
    confidence_score: float = Field(description="A confidence score between 0.0 and 1.0 indicating how well the answer is grounded in the retrieved context.")

def retrieve_node(state: GraphState):
    query = state["query"]
    
    if retriever is None:
        return {"context": []}
        
    docs = retriever.invoke(query)
    context = [doc.page_content for doc in docs]
    return {"context": context}

def generate_node(state: GraphState):
    query = state["query"]
    context = state["context"]
    
    if not context:
        return {
            "answer": "I do not have enough information in the provided document to answer this query.",
            "confidence_score": 0.0
        }
        
    context_str = "\n\n".join(context)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant tasked with answering questions strictly based on the provided context from an eBook. "
                   "If the information is not available in the context, refuse to answer or state that it is not available in the eBook context. "
                   "Provide a confidence score for your answer based on how well the context supports it (0.0 to 1.0).\n"
                   "You must output exactly a JSON object matching this schema:\n"
                   "{{\n"
                   "  \"answer\": \"string\",\n"
                   "  \"confidence_score\": 0.0\n"
                   "}}"),
        ("user", "Context:\n{context}\n\nQuestion: {query}")
    ])
    
    parser = JsonOutputParser(pydantic_object=AnswerFormat)
    
    chain = prompt | llm | parser
    
    response = chain.invoke({
        "context": context_str,
        "query": query
    })
    
    return {
        "answer": response["answer"],
        "confidence_score": float(response["confidence_score"])
    }

# Build Graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app_graph = workflow.compile()
