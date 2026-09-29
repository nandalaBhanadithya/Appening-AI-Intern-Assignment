import requests
import sys

def main():
    query = "What is Agentic AI?"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        
    print(f"\nAsking: '{query}'\n")
    try:
        response = requests.post(
            "http://localhost:8000/query",
            json={"query": query}
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"Answer: {data['final_answer']}")
        print(f"Confidence: {data['confidence_score']}")
        print(f"Retrieved {len(data['retrieved_context_chunks'])} chunks of context.")
        
    except Exception as e:
        print(f"Error querying the API: {e}")

if __name__ == "__main__":
    main()
