"""
SQL Agent for Orion Multi-Agent RAG System
Handles natural language to SQL query conversion and secure sandboxed execution
"""

import os
import httpx
import pandas as pd
import sqlite3
from dotenv import load_dotenv

# Add parent path for imports
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.model_tier import LARGE_TIER



from backend.tools import export_csv

load_dotenv()

# Mistral API configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL_TIER = LARGE_TIER

# Data paths
DATA_PATH_HR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "hr")
DATA_PATH_FINANCE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "finance")

# Schema definitions for the data
HR_SCHEMA = """
TABLE: hr
COLUMNS:
- EmployeeID: string (e.g., "EMP-1001")
- FullName: string (e.g., "Employee_1")
- Department: string (values: HR, Marketing, Sales, Engineering, Product, Finance, Customer Success)
- JobTitle: string (e.g., "HR Coordinator", "Software Engineer", "Manager", "Director", "CFO")
- Location: string (values: Remote, New York, London, Berlin, Tokyo)
- HireDate: date (format: YYYY-MM-DD)
- Salary: integer (range: 50,000 - 160,000)
- PerformanceScore: float (range: 2.5 - 5.0)
- VacationRemaining: integer (range: 5 - 25)
- RemoteStatus: string (values: On-site, Hybrid, Full-time, Remote)
- LastPromotionDate: date or "N/A"
"""

FINANCE_SCHEMA = """
TABLE: finance
COLUMNS:
- transaction_id: string
- date: datetime
- department: string
- category: string (e.g., "Travel", "Training", "Equipment", "Ads")
- amount: float
- created_by: string
- description: string
"""


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


def load_available_data_to_sqlite() -> tuple[sqlite3.Connection, dict]:
    """Load available data sources into an in-memory SQLite database for sandboxed execution"""
    conn = sqlite3.connect(':memory:')
    columns_info = {}
    
    # Load HR data
    try:
        hr_path = os.path.join(DATA_PATH_HR, "HR Anaytics.xlsx")
        if os.path.exists(hr_path):
            df_hr = pd.read_excel(hr_path)
            # Standardize column names if necessary, but here we keep them as is for schema matching
            df_hr.to_sql('hr', conn, index=False, if_exists='replace')
            columns_info['hr'] = list(df_hr.columns)
            print(f"[SQL AGENT] Loaded HR data into SQLite: {df_hr.shape}")
    except Exception as e:
        print(f"[SQL AGENT] Could not load HR data: {e}")
    
    # Load finance data
    try:
        finance_path = os.path.join(DATA_PATH_FINANCE, "Apple Financial.csv")
        if os.path.exists(finance_path):
            df_finance = pd.read_csv(finance_path)
            df_finance.to_sql('finance', conn, index=False, if_exists='replace')
            columns_info['finance'] = list(df_finance.columns)
            print(f"[SQL AGENT] Loaded finance data into SQLite: {df_finance.shape}")
    except Exception as e:
        print(f"[SQL AGENT] Could not load finance data: {e}")
    
    return conn, columns_info


def get_schema_for_llm(columns_info: dict) -> str:
    """Get schema description for LLM using actual column names"""
    schemas = []
    
    if "hr" in columns_info:
        cols = columns_info["hr"]
        schemas.append(f"TABLE 'hr': columns = {cols}")
    
    if "finance" in columns_info:
        cols = columns_info["finance"]
        schemas.append(f"TABLE 'finance': columns = {cols}")
    
    return "\n".join(schemas)


def natural_language_to_sql(question: str, schema: str, feedback: str = "") -> str:
    """Convert natural language query to SQL code"""
    
    feedback_section = f"\nPREVIOUS ERROR TO FIX:\n{feedback}\n" if feedback else ""
    
    prompt = f"""Convert this natural language query to a standard SQLite SQL query.

AVAILABLE DATA:
{schema}

User Query: {question}
{feedback_section}
IMPORTANT RULES:
1. Use ONLY tables and columns that exist in the schema above.
2. Return ONLY the raw SQL code. No markdown, no markdown formatting blocks (like ```sql), and no explanations.
3. The SQL must be immediately executable by sqlite3.
4. If asked for a count, use SELECT COUNT(*).
5. Ensure exact column names are used.

Return ONLY the raw SQL code."""

    try:
        response = call_mistral(prompt)
        # Clean up response
        code = response.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1]) if len(lines) > 2 else lines[0].replace("```", "")
        code = code.replace("```sql", "").replace("```", "").strip()
        return code
    except Exception as e:
        print(f"[SQL AGENT] LLM error: {e}")
        return "SELECT * FROM hr LIMIT 10"


def execute_sql_query(conn: sqlite3.Connection, query_code: str) -> tuple[pd.DataFrame, str]:
    """Safely execute SQL query code in SQLite sandbox"""
    try:
        # read_sql_query safely executes the SQL string against the read-only in-memory DB
        # This is inherently safer than python eval()
        result_df = pd.read_sql_query(query_code, conn)
        return result_df, None
    except Exception as e:
        return None, str(e)


def format_query_result(result: pd.DataFrame, query: str) -> str:
    """Format query result for display"""
    if result is None or result.empty:
        return "No results found."
    
    # Limit display to first 20 rows
    display_df = result.head(20)
    
    # Format numbers nicely
    for col in display_df.select_dtypes(include=['float64']).columns:
        display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
    
    for col in display_df.select_dtypes(include=['int64']).columns:
        display_df[col] = display_df[col].apply(lambda x: f"{x:,}" if pd.notnull(x) else "N/A")
    
    response = f"**Query Results:**\n\n"
    response += display_df.to_markdown(index=False)
    
    if len(result) > 20:
        response += f"\n\n*Showing first 20 of {len(result)} results.*"
    
    return response


def ask_sql(question: str) -> str:
    """
    Main entry point for SQL Agent.
    Handles natural language queries on data using a secure SQLite sandbox
    and an autonomous self-healing LLM loop.
    """
    print("\n" + "="*60)
    print("[SQL AGENT] Processing Request in SQLite Sandbox")
    print("="*60)
    print(f"Query: {question}")
    print("-"*60)
    
    # Load available data into SQLite in-memory sandbox
    conn, columns_info = load_available_data_to_sqlite()
    
    if not columns_info:
        return "No data sources available. Please ensure data files are in the data/EXCEL folder."
    
    # Get schema
    schema = get_schema_for_llm(columns_info)
    print(f"[SQL AGENT] Schema:\n{schema}")
    
    # Autonomous Self-Healing Execution Loop
    max_retries = 3
    feedback = ""
    result = None
    
    for attempt in range(max_retries):
        if feedback:
            print(f"[SQL AGENT] Rewrite Attempt {attempt+1}/{max_retries} due to error...")
            
        # Convert to SQL query
        query_code = natural_language_to_sql(question, schema, feedback)
        print(f"[SQL AGENT] Generated SQL code: {query_code}")
        
        # Execute query in Sandbox
        result, error = execute_sql_query(conn, query_code)
        
        if not error:
            print("[SQL AGENT] Query executed successfully in Sandbox")
            response = format_query_result(result, question)
            print("="*60 + "\n")
            return response
            
        print(f"[SQL AGENT] Query error: {error}")
        feedback = f"The query failed to execute with this SQLite error: {error}. Please fix the SQL syntax or column names and try again."

    # If all retries fail, return a generic safe response
    try:
        table_to_query = list(columns_info.keys())[0]
        fallback_query = f"SELECT * FROM {table_to_query} LIMIT 10"
        result, _ = execute_sql_query(conn, fallback_query)
        response = f"Could not generate an exact query after {max_retries} attempts due to schema constraints. Here is a sample of the data instead:\n\n"
        response += format_query_result(result, question)
        return response
    except:
        return f"Error executing query after {max_retries} attempts."


if __name__ == "__main__":
    # Test the SQL agent sandbox
    print(ask_sql("Show total salary by department"))
    print("\n" + "="*80 + "\n")
    print(ask_sql("Count employees per department"))
