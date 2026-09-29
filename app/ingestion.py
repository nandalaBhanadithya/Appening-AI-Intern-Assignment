import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from app.config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

def ingest_pdf(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")
    
    print(f"Loading PDF from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating index {PINECONE_INDEX_NAME}...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        
    print("Generating embeddings and storing in Pinecone...")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    vectorstore = PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    print("Ingestion complete!")
    return vectorstore

if __name__ == "__main__":
    # Update this path if necessary
    ingest_pdf("data/Ebook-Agentic-AI.pdf")
