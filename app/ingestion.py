import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from app.config import GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

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
    
    # Check if index exists
    existing_indexes = pc.list_indexes().names()
    
    # If index exists, let's make sure it's the right dimension (Gemini is 3072)
    if PINECONE_INDEX_NAME in existing_indexes:
        index_info = pc.describe_index(PINECONE_INDEX_NAME)
        if index_info.dimension != 3072:
            print(f"Deleting existing index {PINECONE_INDEX_NAME} (wrong dimension for Gemini)...")
            pc.delete_index(PINECONE_INDEX_NAME)
            existing_indexes = pc.list_indexes().names()

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating index {PINECONE_INDEX_NAME} (dimension 3072)...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        
    print("Generating embeddings and storing in Pinecone (with rate limit handling)...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=GOOGLE_API_KEY)
    
    vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    
    import time
    # Gemini free tier allows ~100 requests per minute. Langchain batches embeddings.
    # We will upload 80 chunks at a time, then sleep.
    batch_size = 80
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Uploading batch {i//batch_size + 1} of {(len(chunks)-1)//batch_size + 1} ({len(batch)} chunks)...")
        
        retries = 3
        while retries > 0:
            try:
                vectorstore.add_documents(batch)
                break
            except Exception as e:
                print(f"Error encountered: {e}. Sleeping for 60 seconds before retrying...")
                time.sleep(60)
                retries -= 1
        
        if i + batch_size < len(chunks):
            print("Sleeping for 60s to respect Gemini free tier rate limit...")
            time.sleep(60)

    print("Ingestion complete!")
    return vectorstore

if __name__ == "__main__":
    ingest_pdf("data/Ebook-Agentic-AI.pdf")
