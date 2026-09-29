# Agentic AI RAG Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot using LangGraph, Pinecone, and Google Gemini. It strictly answers queries based on the provided document (`Ebook-Agentic-AI.pdf`) and includes groundedness/confidence scoring to prevent hallucinations.

*Note: While the original assignment instructions specified OpenAI, Google Gemini was utilized as a 100% free drop-in replacement API (both for Embeddings and LLM) to avoid API billing costs while fully demonstrating the LangGraph and Pinecone RAG architecture requirements.*

## Architecture

1.  **Data Ingestion Pipeline**:
    *   **PyPDF / LangChain TextSplitter**: Downloads and parses the eBook, splitting text into standard segments (1000 chars with 100 overlap).
    *   **Google Gemini Embeddings + Pinecone**: Converts chunks into dense vectors and indexes them for fast semantic search.
2.  **LangGraph RAG Workflow**:
    *   **Retrieve Node**: Queries the Pinecone vector database for the top-3 most similar chunks.
    *   **Generate Node**: Uses `gemini-1.5-flash` to synthesize a grounded answer based *only* on the retrieved context, and assigns a confidence score.
3.  **API Integration**:
    *   **FastAPI**: Exposes an endpoint (`/query`) returning the requested JSON payload structure containing the answer, context, and score.

## Setup Guide

### 1. Requirements

*   Python 3.9+
*   Google API Key
*   Pinecone API Key

### 2. Installation

Clone this repository and install the dependencies:

```bash
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=agentic-ai-index
```

### 4. Data Ingestion

Place the `Ebook-Agentic-AI.pdf` into the `data/` directory.

Run the ingestion script to parse the PDF, generate embeddings, and store them in Pinecone:

```bash
python -m app.ingestion
```

### 5. Running the API

Start the FastAPI server:

```bash
python -m app.main
```

The server will be available at `http://localhost:8000`. You can access the interactive Swagger UI at `http://localhost:8000/docs`.

### 6. Testing the Endpoint

You can test the endpoint using `curl` or Postman:

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is Agentic AI?"}'
```

## Sample Queries for Validation

1.  **Definition & Scope**: What is the core definition of Agentic AI as outlined in the eBook?
2.  **Architecture & Paradigms**: What are the main architectural components required to build agentic systems?
3.  **Use Cases**: What real-world industry use cases for Agentic AI are discussed in the eBook?
4.  **Comparison**: How does Agentic AI differ from traditional generative AI chatbots according to the text?
5.  **Challenges & Considerations**: What key challenges or limitations of Agentic AI are mentioned in the document?
6.  **Out-of-Scope Test (Groundedness Check)**: What is the capital of France? *(Expected: Refusal to answer, score 0.0)*
