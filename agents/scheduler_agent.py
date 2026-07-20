import os
import httpx
from dotenv import load_dotenv

from agents.model_tier import LARGE_TIER
from vector_store import query_collection

load_dotenv()

COLLECTION_NAME = "scheduler_data"
TOP_K = 8  # Increased for larger calendar corpus (500k events sampled to 50k)

# Mistral API configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL_TIER = LARGE_TIER


def search_events(query: str) -> list[dict]:
    results = query_collection(COLLECTION_NAME, query, top_k=TOP_K)
    return results


def get_events_by_date(date: str) -> str:
    query = f"events on {date}"
    results = search_events(query)
    if results["documents"] and results["documents"][0]:
        return "\n".join(results["documents"][0])
    return "No events found for this date."


def get_events_by_department(department: str) -> str:
    query = f"{department} department meetings events"
    results = search_events(query)
    if results["documents"] and results["documents"][0]:
        return "\n".join(results["documents"][0])
    return f"No events found for {department} department."


def check_availability(date: str, time: str) -> str:
    query = f"meetings scheduled on {date} at {time}"
    results = search_events(query)
    if results["documents"] and results["documents"][0]:
        context = "\n".join(results["documents"][0])
        if time.lower() in context.lower() or date in context:
            return f"There may be conflicts. Existing events:\n{context}"
    return f"No conflicts found for {date} at {time}."


def call_mistral(prompt: str) -> str:
    """Call Mistral API directly via HTTP"""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_TIER,
        "messages": [{"role": "user", "content": prompt}]
    }

    with httpx.Client(timeout=60.0) as client:
        response = client.post(MISTRAL_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


def ask_scheduler(question: str) -> str:
    results = search_events(question)
    context = ""

    # Log retrieved chunks for debugging
    print("\n" + "=" * 60)
    print("[SCHEDULER AGENT] Retrieved Chunks")
    print("=" * 60)
    print(f"Query: {question}")
    print("-" * 60)

    if results["documents"] and results["documents"][0]:
        context = "\n\n".join(results["documents"][0])

        for i, (doc, meta) in enumerate(zip(results["documents"][0],
                                            results["metadatas"][0] if results["metadatas"] and results["metadatas"][
                                                0] else [{}] * len(results["documents"][0])), 1):
            print(f"\nChunk {i}:")
            print(f"  Source: {meta.get('source', 'unknown')}")
            print(f"  Content: {doc[:200]}..." if len(doc) > 200 else f"  Content: {doc}")
    else:
        print("No chunks retrieved!")

    print("=" * 60 + "\n")

    prompt = f"""You are a scheduling assistant. Answer the question based on the calendar data provided.

Calendar Data:
{context}

Question: {question}

Provide a helpful and concise answer."""

    # Log the prompt being sent to LLM
    print("[SCHEDULER AGENT] Prompt sent to LLM:")
    print("-" * 60)
    print(prompt)
    print("-" * 60 + "\n")

    response_text = call_mistral(prompt)

    # Log LLM response
    print("[SCHEDULER AGENT] LLM Response:")
    print("-" * 60)
    print(response_text)
    print("-" * 60 + "\n")

    return response_text