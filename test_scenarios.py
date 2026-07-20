import urllib.request
import json

url = "http://localhost:8000/chat"
headers = {'Content-Type': 'application/json'}

queries = [
    "What is the standard onboarding process for the Marketing department?",
    "Calculate the sum of all training expenses logged in January.",
    "Generate a chart of revenue for Q3 and tell me who the top-performing sales executive is.",
    "Who won the most recent athletic championship?"
]

print("Starting tests...")
for q in queries:
    print(f"\n[{q}]")
    data = json.dumps({"message": q, "stream": False}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            print(f"Response: {result.get('response', '')}")
    except Exception as e:
        print(f"Error: {e}")
