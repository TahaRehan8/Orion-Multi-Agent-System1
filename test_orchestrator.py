import sys
sys.path.insert(0, '.')
from backend.orchestrator import coordinate

queries = [
    "What is the standard onboarding process for the Marketing department?",
    "Calculate the sum of all training expenses logged in January.",
    "Generate a chart of revenue for Q3 and tell me who the top-performing sales executive is.",
    "Who won the most recent athletic championship?"
]

print("Starting tests...")
for q in queries:
    print(f"\n--- QUERY: {q} ---")
    try:
        result = coordinate(q)
        print(f"ACTUAL_RESPONSE:\n{result.final_response}")
        print(f"AGENTS_USED: {result.agents_used}")
    except Exception as e:
        print(f"ERROR: {e}")
