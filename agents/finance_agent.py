"""
Finance Agent for Orion Multi-Agent RAG System
Handles financial data queries with ARIMA forecasting and anomaly detection
"""

import os
import httpx
import pandas as pd
from dotenv import load_dotenv

# Add parent path for imports
import sys

from agents.model_tier import LARGE_TIER

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import query_collection
from backend.tools import arima_forecast, anomaly_detection, get_agent_tools

load_dotenv()

COLLECTION_NAME = "finance_data"
TOP_K = 7  # Increased: larger corpus with SEC filings needs more context chunks

# Mistral API configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL_TIER = LARGE_TIER

# Data path for finance data
# Real Apple financial CSV (used for ARIMA and anomaly tools)
FINANCE_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "finance", "ApplFin.csv")


def search_documents(query: str) -> list[dict]:
    results = query_collection(COLLECTION_NAME, query, top_k=TOP_K)
    return results


def search_by_topic(topic: str) -> str:
    results = search_documents(topic)
    if results["documents"] and results["documents"][0]:
        return "\n\n---\n\n".join(results["documents"][0])
    return f"No documents found for topic: {topic}"


def get_financial_metrics(metric_type: str) -> str:
    query = f"{metric_type} financial data numbers statistics"
    results = search_documents(query)
    if results["documents"] and results["documents"][0]:
        return "\n\n".join(results["documents"][0])
    return f"No data found for {metric_type}."


def identify_anomalies(area: str = "general") -> str:
    query = f"variance anomaly unusual {area} issue problem"
    results = search_documents(query)
    if results["documents"] and results["documents"][0]:
        return "\n\n".join(results["documents"][0])
    return "No anomalies identified."


def get_forecast_data() -> str:
    query = "forecast prediction future projection trend"
    results = search_documents(query)
    if results["documents"] and results["documents"][0]:
        return "\n\n".join(results["documents"][0])
    return "No forecast data available."


def load_finance_data() -> pd.DataFrame:
    """Load finance data for ARIMA/anomaly tools from ApplFin.csv."""
    try:
        if os.path.exists(FINANCE_DATA_PATH):
            return pd.read_csv(FINANCE_DATA_PATH)
    except Exception as e:
        print(f"[FINANCE AGENT] Could not load finance CSV data: {e}")
    return pd.DataFrame()


def call_mistral(prompt: str) -> str:
    """Call Mistral API directly via HTTP"""
    import time
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_TIER,
        "messages": [{"role": "user", "content": prompt}]
    }

    max_retries = 6
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(MISTRAL_API_URL, json=payload, headers=headers)
                if response.status_code == 429 and attempt < max_retries - 1:
                    print(f"[FINANCE AGENT] 429 Rate Limit, waiting 30s...")
                    time.sleep(30)
                    continue
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"[FINANCE AGENT] API Error: {e}, retrying in 30s...")
            time.sleep(30)


def detect_tool_need(question: str) -> str:
    """Detect if a specific tool should be used based on the question"""
    question_lower = question.lower()

    # Check for forecasting keywords
    if any(kw in question_lower for kw in ["forecast", "predict", "projection", "next month", "future", "arima"]):
        return "arima_forecast"

    # Check for anomaly detection keywords
    if any(kw in question_lower for kw in ["anomaly", "anomalies", "outlier", "unusual", "abnormal", "variance"]):
        return "anomaly_detection"

    return None


def run_arima_tool(df: pd.DataFrame, question: str) -> str:
    """Run ARIMA forecasting on finance data"""
    # Try to find a numeric column for forecasting
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    if len(numeric_cols) == 0:
        return "No numeric data available for forecasting."

    # Use the first numeric column or try to find revenue/expense column
    target_col = None
    for col in numeric_cols:
        if any(kw in col.lower() for kw in ["revenue", "amount", "total", "value", "expense"]):
            target_col = col
            break

    if target_col is None:
        target_col = numeric_cols[0]

    data_series = df[target_col].dropna()

    if len(data_series) < 10:
        return f"Insufficient data for ARIMA forecasting (only {len(data_series)} data points available, need at least 10)."

    result = arima_forecast(data_series, periods=3)

    if result.get("success"):
        forecast_values = result.get("forecast", [])
        return f"""**ARIMA Forecast Results for {target_col}:**

Forecasted values for next 3 periods:
- Period 1: {forecast_values[0]:.2f}
- Period 2: {forecast_values[1]:.2f}
- Period 3: {forecast_values[2]:.2f}

Model Quality:
- AIC: {result.get('aic', 'N/A'):.2f}
- BIC: {result.get('bic', 'N/A'):.2f}

Note: These forecasts are based on historical trends in the {target_col} data."""
    else:
        return f"Forecasting error: {result.get('error', 'Unknown error')}"


def run_anomaly_tool(df: pd.DataFrame, question: str) -> str:
    """Run anomaly detection on finance data"""
    # Try to find a numeric column for anomaly detection
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    if len(numeric_cols) == 0:
        return "No numeric data available for anomaly detection."

    results = []
    for col in numeric_cols[:3]:  # Check up to 3 numeric columns
        data_series = df[col].dropna()

        if len(data_series) < 5:
            continue

        result = anomaly_detection(data_series, threshold=2.0)

        if result.get("success") and result.get("anomaly_count", 0) > 0:
            anomalies = result.get("anomalies", {})
            results.append(f"""**Anomalies in {col}:**
- Count: {result.get('anomaly_count')} anomalies found
- Mean: {result.get('mean', 0):.2f}
- Std Dev: {result.get('std', 0):.2f}
- Anomalous values: {list(anomalies.values())[:5]}""")

    if results:
        return "**Anomaly Detection Results:**\n\n" + "\n\n".join(results)
    else:
        return "No significant anomalies detected in the financial data. All values are within normal ranges (within 2 standard deviations of the mean)."


def ask_finance(question: str) -> str:
    """Main entry point for Finance Agent with tool integration"""

    print("\n" + "="*60)
    print("[FINANCE AGENT] Processing Request")
    print("="*60)
    print(f"Query: {question}")
    print("-"*60)

    # Check if a tool should be used
    tool_needed = detect_tool_need(question)

    if tool_needed:
        print(f"[FINANCE AGENT] Tool detected: {tool_needed}")
        df = load_finance_data()

        if not df.empty:
            if tool_needed == "arima_forecast":
                tool_result = run_arima_tool(df, question)
                print(f"[FINANCE AGENT] ARIMA tool executed")
                print("="*60 + "\n")
                return tool_result

            elif tool_needed == "anomaly_detection":
                tool_result = run_anomaly_tool(df, question)
                print(f"[FINANCE AGENT] Anomaly detection tool executed")
                print("="*60 + "\n")
                return tool_result

    # Standard RAG flow
    results = search_documents(question)
    context = ""
    sources = []

    print("[FINANCE AGENT] Using RAG flow")

    if results["documents"] and results["documents"][0]:
        context = "\n\n".join(results["documents"][0])
        if results["metadatas"] and results["metadatas"][0]:
            sources = [m.get("source", "unknown") for m in results["metadatas"][0]]

        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0] if results["metadatas"] and results["metadatas"][0] else [{}]*len(results["documents"][0])), 1):
            print(f"\nChunk {i}:")
            print(f"  Source: {meta.get('source', 'unknown')}")
            print(f"  Content: {doc[:200]}..." if len(doc) > 200 else f"  Content: {doc}")
    else:
        print("No chunks retrieved!")

    print("="*60 + "\n")

    prompt = f"""You are a finance analyst assistant. Answer the question based on the financial data provided.

Financial Data:
{context}

Sources: {', '.join(set(sources)) if sources else 'Various documents'}

Question: {question}

Provide a helpful, accurate, and concise answer. Cite specific numbers when available."""

    print("[FINANCE AGENT] Prompt sent to LLM:")
    print("-"*60)
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    print("-"*60 + "\n")

    response_text = call_mistral(prompt)

    print("[FINANCE AGENT] LLM Response:")
    print("-"*60)
    print(response_text)
    print("-"*60 + "\n")

    return response_text


if __name__ == "__main__":
    # Test the finance agent
    print("\n--- Testing Finance Agent ---\n")
    print(ask_finance("Forecast next 3 months revenue"))
    print("\n" + "="*80 + "\n")
    print(ask_finance("Are there any anomalies in expenses?"))