"""
Orion — Knowledge Update Test Suite
====================================
Validates the new real-data vector DB across:
  1. Collection existence and population
  2. Retrieval latency (p50 / p95) per agent
  3. HR retrieval accuracy (factual queries, LLM-as-judge)
  4. Finance retrieval accuracy (SEC / FRED queries, LLM-as-judge)
  5. Scheduler retrieval accuracy (date / organizer queries, LLM-as-judge)
  6. Cross-collection coverage (SEC filings, policy PDF, MNC HR, products)

Results written to: knowledge_update_test_results.json

Usage:
    python test_knowledge_update.py
"""

import os
import sys
import json
import time
import httpx
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
RESULTS_FILE = os.path.join(PROJECT_ROOT, "knowledge_update_test_results.json")

# Delay between Mistral API calls to respect rate limits
API_DELAY = 8

# ── Helpers ────────────────────────────────────────────────────────────────────

def _call_mistral(prompt: str, model: str = "mistral-large-latest") -> str:
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    max_retries = 4
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=90.0) as client:
                resp = client.post(MISTRAL_API_URL, json=payload, headers=headers)
                if resp.status_code == 429:
                    wait = float(resp.headers.get("Retry-After", min(2 ** attempt, 30)))
                    print(f"    [judge] Rate-limited — waiting {wait:.0f}s…")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == max_retries - 1:
                return f"ERROR: {e}"
            time.sleep(min(2 ** attempt, 15))
    return "ERROR: max retries exceeded"


def _extract_json(text: str) -> dict:
    for delimiter in [("```json", "```"), ("```", "```")]:
        if delimiter[0] in text:
            text = text.split(delimiter[0], 1)[1].split(delimiter[1], 1)[0]
            break
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return {}


def _judge_response(query: str, response: str, ground_truth_hint: str = "") -> dict:
    """Use Mistral-Large as judge — returns {score: 0-100, reasoning: str}."""
    gt_line = f"GROUND TRUTH HINT: {ground_truth_hint}\n" if ground_truth_hint else ""
    prompt = f"""You are an expert evaluator for an enterprise AI system.
Score the AI response from 0 to 100 based on how well it answers the query.

QUERY: {query}
{gt_line}AI RESPONSE: {response[:1200]}

Scoring criteria:
- Factual accuracy and relevance to the query
- Use of specific data (numbers, names, dates) from the knowledge base
- No hallucinated or invented information
- Concise and complete answer

Return ONLY JSON: {{"score": <0-100>, "reasoning": "<brief explanation>"}}"""

    raw = _call_mistral(prompt)
    parsed = _extract_json(raw)
    return {
        "score": int(parsed.get("score", 0)),
        "reasoning": str(parsed.get("reasoning", "parse error")),
    }


def _percentile(data: list[float], p: int) -> float:
    arr = sorted(data)
    if not arr:
        return 0.0
    idx = (p / 100) * (len(arr) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(arr) - 1)
    return arr[lo] + (arr[hi] - arr[lo]) * (idx - lo)


# ── Test 1: Collection Existence & Population ──────────────────────────────────

def test_collection_existence() -> dict:
    print("\n" + "=" * 60)
    print("TEST 1: Collection Existence & Population")
    print("=" * 60)
    from vector_store import get_collection_count

    collections = ["finance_data", "hr_data", "scheduler_data"]
    results = {}
    all_pass = True

    for col in collections:
        count = get_collection_count(col)
        passed = count > 0
        if not passed:
            all_pass = False
        status = "✅ PASS" if passed else "❌ FAIL (empty)"
        print(f"  {col}: {count:,} documents  {status}")
        results[col] = {"count": count, "passed": passed}

    print(f"\n  Overall: {'✅ ALL POPULATED' if all_pass else '❌ SOME EMPTY'}")
    return {"test": "Collection Existence", "passed": all_pass, "details": results}


# ── Test 2: Retrieval Latency (p50 / p95) ─────────────────────────────────────

def test_retrieval_latency() -> dict:
    print("\n" + "=" * 60)
    print("TEST 2: Retrieval Latency (p50 / p95)")
    print("=" * 60)
    from vector_store import query_collection

    latency_queries = {
        "finance_data": [
            "Apple revenue fiscal year 2023",
            "Risk factors Apple 10-K",
            "Federal funds rate 2022",
            "Consumer Price Index inflation",
            "Apple iPhone sales revenue",
            "SEC filing management discussion analysis",
            "Operating expenses Apple quarterly",
            "Consumer sentiment economic outlook",
            "Apple gross margin 2024",
            "AAPL dividend per share",
        ],
        "hr_data": [
            "Engineering department employees",
            "Performance rating excellent",
            "Employees in New York remote work",
            "Data science job title salary",
            "Corporate ethics policy",
            "High performance employees Finance department",
            "Employees hired 2023",
            "Salary above 100000",
            "Business conduct policy compliance",
            "HR manager location",
        ],
        "scheduler_data": [
            "Q4 planning meeting",
            "Board meeting January 2026",
            "Finance review scheduled events",
            "Team standup organizer",
            "Conference call weekly",
            "Product launch event",
            "Budget review calendar",
            "HR meeting attendees",
            "Engineering sprint review",
            "Customer presentation schedule",
        ],
    }

    TARGET_P95_MS = 2000  # 2 second target
    all_results = {}
    overall_pass = True

    for collection, queries in latency_queries.items():
        latencies_ms = []
        for q in queries:
            t0 = time.perf_counter()
            try:
                query_collection(collection, q, top_k=5)
            except Exception:
                pass
            latencies_ms.append((time.perf_counter() - t0) * 1000)

        p50 = _percentile(latencies_ms, 50)
        p95 = _percentile(latencies_ms, 95)
        passed = p95 <= TARGET_P95_MS
        if not passed:
            overall_pass = False

        status = "✅ PASS" if passed else "❌ SLOW"
        print(f"  {collection}: p50={p50:.0f}ms  p95={p95:.0f}ms  (target p95<{TARGET_P95_MS}ms) {status}")
        all_results[collection] = {
            "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1),
            "target_p95_ms": TARGET_P95_MS,
            "passed": passed,
            "latencies_ms": [round(l, 1) for l in latencies_ms],
        }

    return {"test": "Retrieval Latency", "passed": overall_pass, "details": all_results}


# ── Test 3: HR Retrieval Accuracy ──────────────────────────────────────────────

def test_hr_retrieval_accuracy() -> dict:
    print("\n" + "=" * 60)
    print("TEST 3: HR Retrieval Accuracy (LLM-as-Judge)")
    print("=" * 60)
    from agents.hr_agent import ask_hr

    cases = [
        {
            "query": "What departments exist in the company?",
            "hint": "Should list multiple departments like Engineering, Finance, HR, Marketing, Sales, etc.",
        },
        {
            "query": "What is the business conduct policy on conflicts of interest?",
            "hint": "Should reference the Business Conduct Policy PDF content about ethics or conflicts of interest.",
        },
        {
            "query": "Show me employees who work in a Data Science role.",
            "hint": "Should return employee names and details with 'Data Science' in their job title.",
        },
        {
            "query": "What are the possible work modes for employees?",
            "hint": "Expected: Remote, Hybrid, On-site, or similar work mode categories.",
        },
        {
            "query": "List employees with Excellent performance rating.",
            "hint": "Should return employee records showing Performance_Rating = Excellent.",
        },
    ]

    scores = []
    details = []

    for i, case in enumerate(cases):
        print(f"\n  [{i+1}/{len(cases)}] {case['query']}")
        try:
            response = ask_hr(case["query"])
            time.sleep(API_DELAY)
            judgment = _judge_response(case["query"], response, case["hint"])
            score = judgment["score"]
            print(f"    Score: {score}/100 — {judgment['reasoning'][:80]}")
        except Exception as e:
            score = 0
            judgment = {"score": 0, "reasoning": str(e)}
            print(f"    ❌ ERROR: {e}")
        scores.append(score)
        details.append({**case, "score": score, "reasoning": judgment["reasoning"]})
        time.sleep(API_DELAY)

    avg = sum(scores) / len(scores) if scores else 0
    passed = avg >= 65  # reasonable threshold for large diverse corpus
    print(f"\n  HR Avg Score: {avg:.1f}/100  {'✅ PASS' if passed else '❌ BELOW THRESHOLD'}")
    return {"test": "HR Retrieval Accuracy", "avg_score": round(avg, 1), "target": 65, "passed": passed, "details": details}


# ── Test 4: Finance Retrieval Accuracy ────────────────────────────────────────

def test_finance_retrieval_accuracy() -> dict:
    print("\n" + "=" * 60)
    print("TEST 4: Finance Retrieval Accuracy (LLM-as-Judge)")
    print("=" * 60)
    from agents.finance_agent import ask_finance

    cases = [
        {
            "query": "What are the main risk factors disclosed in Apple's 10-K SEC filing?",
            "hint": "Should mention specific risk categories from Apple's annual report (supply chain, competition, regulation, etc.)",
        },
        {
            "query": "What was the trend in Consumer Price Index (CPI) in 2022?",
            "hint": "Should reference FRED CPI data showing inflation trends in 2022.",
        },
        {
            "query": "What product categories does Apple sell?",
            "hint": "Should describe iPhone, iPad, Mac, Apple Watch, services, etc. from product dataset.",
        },
        {
            "query": "What does Apple's management say about iPhone revenue in their quarterly reports?",
            "hint": "Should reference 10-Q SEC filing content about iPhone product revenue.",
        },
        {
            "query": "What is the Federal Funds effective rate trend?",
            "hint": "Should reference FRED DFF data showing interest rate changes.",
        },
    ]

    scores = []
    details = []

    for i, case in enumerate(cases):
        print(f"\n  [{i+1}/{len(cases)}] {case['query']}")
        try:
            response = ask_finance(case["query"])
            time.sleep(API_DELAY)
            judgment = _judge_response(case["query"], response, case["hint"])
            score = judgment["score"]
            print(f"    Score: {score}/100 — {judgment['reasoning'][:80]}")
        except Exception as e:
            score = 0
            judgment = {"score": 0, "reasoning": str(e)}
            print(f"    ❌ ERROR: {e}")
        scores.append(score)
        details.append({**case, "score": score, "reasoning": judgment["reasoning"]})
        time.sleep(API_DELAY)

    avg = sum(scores) / len(scores) if scores else 0
    passed = avg >= 65
    print(f"\n  Finance Avg Score: {avg:.1f}/100  {'✅ PASS' if passed else '❌ BELOW THRESHOLD'}")
    return {"test": "Finance Retrieval Accuracy", "avg_score": round(avg, 1), "target": 65, "passed": passed, "details": details}


# ── Test 5: Scheduler Retrieval Accuracy ──────────────────────────────────────

def test_scheduler_retrieval_accuracy() -> dict:
    print("\n" + "=" * 60)
    print("TEST 5: Scheduler Retrieval Accuracy (LLM-as-Judge)")
    print("=" * 60)
    from agents.scheduler_agent import ask_scheduler

    cases = [
        {
            "query": "What events are scheduled in January 2026?",
            "hint": "Should return calendar events from the 2026 ICS calendar.",
        },
        {
            "query": "Find meetings organized by a finance team member.",
            "hint": "Should return event entries showing Finance-related organizers or titles.",
        },
        {
            "query": "Are there any confidential calendar events?",
            "hint": "Should identify events marked confidential=True in the calendar JSON.",
        },
        {
            "query": "What recurring events involve multiple attendees?",
            "hint": "Should return events with multiple attendees listed.",
        },
        {
            "query": "What HR-related events are on the calendar?",
            "hint": "Should return events with HR in the title or description.",
        },
    ]

    scores = []
    details = []

    for i, case in enumerate(cases):
        print(f"\n  [{i+1}/{len(cases)}] {case['query']}")
        try:
            response = ask_scheduler(case["query"])
            time.sleep(API_DELAY)
            judgment = _judge_response(case["query"], response, case["hint"])
            score = judgment["score"]
            print(f"    Score: {score}/100 — {judgment['reasoning'][:80]}")
        except Exception as e:
            score = 0
            judgment = {"score": 0, "reasoning": str(e)}
            print(f"    ❌ ERROR: {e}")
        scores.append(score)
        details.append({**case, "score": score, "reasoning": judgment["reasoning"]})
        time.sleep(API_DELAY)

    avg = sum(scores) / len(scores) if scores else 0
    passed = avg >= 60  # slightly lower threshold: sampled calendar data
    print(f"\n  Scheduler Avg Score: {avg:.1f}/100  {'✅ PASS' if passed else '❌ BELOW THRESHOLD'}")
    return {"test": "Scheduler Retrieval Accuracy", "avg_score": round(avg, 1), "target": 60, "passed": passed, "details": details}


# ── Test 6: Cross-Collection Data Coverage ────────────────────────────────────

def test_data_coverage() -> dict:
    print("\n" + "=" * 60)
    print("TEST 6: Cross-Collection Data Coverage")
    print("=" * 60)
    from vector_store import query_collection

    coverage_checks = [
        {
            "name": "SEC 10-K filings present",
            "collection": "finance_data",
            "query": "Apple 10-K annual report risk factors",
            "check_meta_type": "sec_filing",
        },
        {
            "name": "FRED macro data present",
            "collection": "finance_data",
            "query": "Consumer Price Index CPI Federal Funds Rate",
            "check_meta_type": "fred_macro",
        },
        {
            "name": "Product catalog present",
            "collection": "finance_data",
            "query": "iPhone iPad Mac storage RAM price category",
            "check_meta_type": "product_catalog",
        },
        {
            "name": "MNC HR employee records present",
            "collection": "hr_data",
            "query": "Employee_ID Department Job_Title Salary_INR",
            "check_meta_type": "hr_employee",
        },
        {
            "name": "HR policy document present",
            "collection": "hr_data",
            "query": "business conduct policy ethics compliance",
            "check_meta_type": "hr_policy",
        },
        {
            "name": "JSON calendar events present",
            "collection": "scheduler_data",
            "query": "event organizer attendees start_time location",
            "check_meta_type": "calendar_event",
        },
        {
            "name": "ICS calendar events present",
            "collection": "scheduler_data",
            "query": "2026 meeting scheduled event",
            "check_meta_type": "calendar_ics",
        },
    ]

    results = []
    passed_count = 0

    for check in coverage_checks:
        try:
            result = query_collection(check["collection"], check["query"], top_k=10)
            metas = result.get("metadatas", [[]])[0]
            found_types = [m.get("type", "") for m in metas]
            found_target = check["check_meta_type"] in found_types
            if found_target:
                passed_count += 1
            status = "✅ FOUND" if found_target else "❌ NOT FOUND"
            print(f"  {check['name']}: {status}")
            results.append({
                "name": check["name"],
                "found": found_target,
                "types_retrieved": list(set(found_types)),
            })
        except Exception as e:
            print(f"  {check['name']}: ❌ ERROR — {e}")
            results.append({"name": check["name"], "found": False, "error": str(e)})

    coverage_pct = (passed_count / len(coverage_checks)) * 100
    passed = coverage_pct >= 85
    print(f"\n  Coverage: {passed_count}/{len(coverage_checks)} = {coverage_pct:.0f}%  {'✅ PASS' if passed else '❌ BELOW THRESHOLD'}")
    return {
        "test": "Data Coverage",
        "coverage_pct": round(coverage_pct, 1),
        "passed_count": passed_count,
        "total": len(coverage_checks),
        "passed": passed,
        "details": results,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("Orion — Knowledge Update Test Suite")
    print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_results = {}
    suite_start = time.time()

    # Tests that don't require LLM API calls — run first for fast feedback
    all_results["collection_existence"] = test_collection_existence()
    all_results["retrieval_latency"] = test_retrieval_latency()
    all_results["data_coverage"] = test_data_coverage()

    # LLM-judge tests — run sequentially to respect rate limits
    all_results["hr_accuracy"] = test_hr_retrieval_accuracy()
    all_results["finance_accuracy"] = test_finance_retrieval_accuracy()
    all_results["scheduler_accuracy"] = test_scheduler_retrieval_accuracy()

    # ── Summary ──
    total_time = time.time() - suite_start
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for key, result in all_results.items():
        p = result.get("passed", False)
        if not p:
            all_passed = False
        # Find the main metric
        metric = ""
        if "avg_score" in result:
            metric = f"avg={result['avg_score']}/100"
        elif "coverage_pct" in result:
            metric = f"coverage={result['coverage_pct']}%"
        elif "p95_ms" in result.get("details", {}).get("finance_data", {}):
            metric = "see latency details"
        status = "✅" if p else "❌"
        print(f"  {status} {result['test']}  {metric}")

    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  Overall: {'✅ ALL TESTS PASSED' if all_passed else '⚠️ SOME TESTS NEED ATTENTION'}")

    # Save results
    all_results["meta"] = {
        "timestamp": datetime.now().isoformat(),
        "total_time_s": round(total_time, 1),
        "overall_passed": all_passed,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
