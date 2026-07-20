"""
Tools Module for Orion Multi-Agent RAG System
Provides tools for ARIMA forecasting, anomaly detection, graph generation, and data export
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Audit logging (fail-safe)
try:
    from backend.audit import log_event
except ImportError:
    def log_event(*a, **kw): pass

# Output directories
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
EXPORTS_DIR = OUTPUT_DIR / "exports"
GRAPHS_DIR = OUTPUT_DIR / "graphs"

# Create directories
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# Agent-specific tools registry
AGENT_TOOLS = {
    "finance": ["arima_forecast", "anomaly_detection", "generate_graph", "export_csv", "export_notes"],
    "hr": ["generate_graph", "export_csv", "export_notes"],
    "scheduler": ["export_csv", "export_notes"],
    "chart": ["generate_graph", "export_csv"],
    "sql": ["execute_query", "export_csv"]
}


def get_agent_tools(agent_name: str) -> List[str]:
    """Get list of available tools for an agent"""
    return AGENT_TOOLS.get(agent_name, [])


# ============ ARIMA Forecasting ============

def arima_forecast(data: pd.Series, periods: int = 3) -> Dict[str, Any]:
    """
    Perform ARIMA forecasting on time series data.
    
    Args:
        data: Time series data as pandas Series
        periods: Number of periods to forecast
        
    Returns:
        Dictionary with forecast results
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        
        # Clean data
        data = data.dropna()
        if len(data) < 10:
            return {
                "success": False,
                "error": "Insufficient data for ARIMA forecast (need at least 10 data points)"
            }
        
        # Fit ARIMA model (1,1,1) as default
        model = ARIMA(data, order=(1, 1, 1))
        fitted = model.fit()
        
        # Forecast
        forecast = fitted.forecast(steps=periods)
        
        return {
            "success": True,
            "forecast": forecast.tolist(),
            "periods": periods,
            "model_summary": str(fitted.summary()),
            "aic": fitted.aic,
            "bic": fitted.bic
        }
        
    except ImportError:
        return {
            "success": False,
            "error": "statsmodels not installed. Run: pip install statsmodels"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"ARIMA forecast failed: {str(e)}"
        }


# ============ Anomaly Detection ============

def anomaly_detection(data: pd.Series, threshold: float = 2.0) -> Dict[str, Any]:
    """
    Detect anomalies in time series using Z-score method.
    
    Args:
        data: Time series data as pandas Series
        threshold: Z-score threshold for anomaly detection (default 2.0)
        
    Returns:
        Dictionary with anomaly detection results
    """
    try:
        data = data.dropna()
        if len(data) < 5:
            return {
                "success": False,
                "error": "Insufficient data for anomaly detection (need at least 5 data points)"
            }
        
        # Calculate Z-scores
        mean = data.mean()
        std = data.std()
        
        if std == 0:
            return {
                "success": True,
                "anomalies": [],
                "message": "No variance in data - no anomalies possible"
            }
        
        z_scores = np.abs((data - mean) / std)
        anomaly_mask = z_scores > threshold
        anomalies = data[anomaly_mask]
        
        return {
            "success": True,
            "anomalies": anomalies.to_dict(),
            "anomaly_count": len(anomalies),
            "total_points": len(data),
            "threshold": threshold,
            "mean": float(mean),
            "std": float(std)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Anomaly detection failed: {str(e)}"
        }


# ============ Graph Generation ============

def generate_graph(
    data: pd.DataFrame,
    graph_type: str = "bar",
    title: str = "Chart",
    x_col: str = None,
    y_col: str = None,
    save: bool = True,
    aggregate: bool = True,
    max_categories: int = 15
) -> Dict[str, Any]:
    """
    Generate graph using plotly with proper data aggregation.
    
    Args:
        data: DataFrame with data to plot
        graph_type: Type of graph (bar, line, pie, scatter)
        title: Chart title
        x_col: Column for x-axis (category/grouping)
        y_col: Column for y-axis (values to aggregate)
        save: Whether to save as image
        aggregate: Whether to aggregate data by x_col
        max_categories: Maximum number of categories to show (prevents cluttered charts)
        
    Returns:
        Dictionary with graph JSON and optional file path
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        
        if data.empty:
            return {
                "success": False,
                "error": "Empty data provided"
            }
        
        # Auto-detect columns if not specified
        numeric_cols = data.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
        non_numeric_cols = [c for c in data.columns if c not in numeric_cols]
        
        if x_col is None:
            # Prefer categorical columns for x-axis
            x_col = non_numeric_cols[0] if non_numeric_cols else data.columns[0]
        
        if y_col is None:
            # Prefer numeric columns for y-axis
            y_col = numeric_cols[0] if numeric_cols else (data.columns[1] if len(data.columns) > 1 else data.columns[0])
        
        # Validate columns exist
        if x_col not in data.columns:
            x_col = data.columns[0]
            
        is_count = str(y_col).lower() == 'count'
        
        if not is_count and y_col not in data.columns:
            y_col = numeric_cols[0] if numeric_cols else data.columns[-1]
        
        # AGGREGATE DATA - This is the key fix!
        if aggregate and x_col != y_col:
            # Group by x_col and sum/count y_col
            if is_count:
                # Explicit count request
                plot_data = data.groupby(x_col).size().reset_index(name='count')
                y_col = 'count'
            elif y_col in numeric_cols:
                # For numeric y, aggregate by sum
                plot_data = data.groupby(x_col, as_index=False)[y_col].sum()
            else:
                # For non-numeric y, count occurrences
                plot_data = data.groupby(x_col).size().reset_index(name='count')
                y_col = 'count'
            
            # Sort by value and limit categories
            plot_data = plot_data.sort_values(y_col, ascending=False).head(max_categories)
        else:
            plot_data = data.head(max_categories)
        
        # Clean up data for display
        plot_data = plot_data.copy()
        
        # Shorten long category labels
        if plot_data[x_col].dtype == 'object':
            plot_data[x_col] = plot_data[x_col].astype(str).apply(
                lambda x: x[:25] + '...' if len(str(x)) > 25 else x
            )
        
        print(f"[GRAPH] Plotting {len(plot_data)} rows, x={x_col}, y={y_col}")
        
        # Create figure based on type
        if graph_type == "line":
            fig = px.line(
                plot_data, x=x_col, y=y_col, title=title,
                markers=True
            )
        elif graph_type == "bar":
            fig = px.bar(
                plot_data, x=x_col, y=y_col, title=title,
                text=y_col  # Show values on bars
            )
            fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        elif graph_type == "pie":
            fig = px.pie(
                plot_data, names=x_col, values=y_col, title=title,
                hole=0.3  # Donut style for better readability
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
        elif graph_type == "scatter":
            fig = px.scatter(
                plot_data, x=x_col, y=y_col, title=title,
                size=y_col if y_col in numeric_cols else None
            )
        else:
            fig = px.bar(plot_data, x=x_col, y=y_col, title=title)
        
        # Apply clean dark theme
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1e293b",
            plot_bgcolor="#0f172a",
            font=dict(size=12),
            title=dict(font=dict(size=18)),
            margin=dict(l=60, r=40, t=60, b=80),
            showlegend=True if graph_type == "pie" else False
        )
        
        # Better axis formatting
        if graph_type in ["bar", "line", "scatter"]:
            fig.update_xaxes(tickangle=45, title=x_col.replace("_", " ").title())
            fig.update_yaxes(title=y_col.replace("_", " ").title())
        
        result = {
            "success": True,
            "graph_json": fig.to_json(),
            "graph_type": graph_type,
            "data_points": len(plot_data),
            "x_column": x_col,
            "y_column": y_col
        }
        
        # Save as image if requested
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"graph_{graph_type}_{timestamp}.png"
            filepath = GRAPHS_DIR / filename
            
            try:
                fig.write_image(str(filepath), width=800, height=500, scale=2)
                result["file_path"] = str(filepath)
            except Exception as e:
                result["save_error"] = f"Could not save image: {str(e)}"
        
        return result
        
    except ImportError:
        return {
            "success": False,
            "error": "plotly not installed. Run: pip install plotly"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Graph generation failed: {str(e)}"
        }


# ============ Data Export ============

def export_csv(data: pd.DataFrame, filename: str = None) -> Dict[str, Any]:
    """
    Export data to CSV file.
    
    Args:
        data: DataFrame to export
        filename: Optional filename (auto-generated if not provided)
        
    Returns:
        Dictionary with export result
    """
    try:
        if data.empty:
            return {
                "success": False,
                "error": "Empty data provided"
            }
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"export_{timestamp}.csv"
        
        if not filename.endswith(".csv"):
            filename += ".csv"
        
        filepath = EXPORTS_DIR / filename
        data.to_csv(filepath, index=False)
        
        return {
            "success": True,
            "file_path": str(filepath),
            "rows": len(data),
            "columns": list(data.columns)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"CSV export failed: {str(e)}"
        }


def export_notes(text: str, filename: str = None) -> Dict[str, Any]:
    """
    Export text notes to TXT file.
    
    Args:
        text: Text content to export
        filename: Optional filename (auto-generated if not provided)
        
    Returns:
        Dictionary with export result
    """
    try:
        if not text or not text.strip():
            return {
                "success": False,
                "error": "Empty text provided"
            }
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"notes_{timestamp}.txt"
        
        if not filename.endswith(".txt"):
            filename += ".txt"
        
        filepath = EXPORTS_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        
        return {
            "success": True,
            "file_path": str(filepath),
            "characters": len(text)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Notes export failed: {str(e)}"
        }


# ─────────────────────────────────────────────
# RBAC: tools allowed per user role
# ─────────────────────────────────────────────

ROLE_TOOL_ALLOWLIST: Dict[str, List[str]] = {
    "limited": ["generate_graph", "export_csv", "export_notes", "execute_query", "anomaly_detection"],
    "admin":   ["arima_forecast", "anomaly_detection", "generate_graph", "export_csv", "export_notes", "execute_query"],
}
# Super-user and None (internal/system) can use all tools
_ALL_TOOLS = list(AGENT_TOOLS["finance"]) + ["execute_query"]


def _validate_tool_schema(tool_name: str, kwargs: dict) -> Optional[str]:
    """
    Basic parameter validation per tool.
    Returns an error string on failure, None on success.
    """
    if tool_name == "arima_forecast":
        data = kwargs.get("data")
        periods = kwargs.get("periods", 3)
        if data is not None and not isinstance(data, pd.Series):
            return "arima_forecast: 'data' must be a pandas Series"
        if not isinstance(periods, int) or not (1 <= periods <= 24):
            return f"arima_forecast: 'periods' must be int in [1, 24], got {periods!r}"

    elif tool_name == "generate_graph":
        data = kwargs.get("data")
        graph_type = kwargs.get("graph_type", "bar")
        allowed_types = ["bar", "line", "pie", "scatter"]
        if data is not None and not isinstance(data, pd.DataFrame):
            return "generate_graph: 'data' must be a pandas DataFrame"
        if graph_type not in allowed_types:
            return f"generate_graph: 'graph_type' must be one of {allowed_types}, got {graph_type!r}"

    return None  # OK


def execute_tool(
    tool_name: str,
    user_role: Optional[str] = None,
    request_id: str = "ORION-INTERNAL",
    user: str = "system",
    **kwargs,
) -> Dict[str, Any]:
    """
    Execute a tool by name with provided arguments.
    Enforces RBAC and validates parameters before execution.

    Args:
        tool_name:  Name of the tool to execute
        user_role:  Caller role ('admin', 'limited', None=internal/system)
        request_id: Audit correlation ID
        user:       Requesting username
        **kwargs:   Arguments to pass to the tool

    Returns:
        Tool execution result dict
    """
    # ── RBAC check ────────────────────────────────────────────────────────
    if user_role is not None:
        allowed = ROLE_TOOL_ALLOWLIST.get(user_role, [])
        if tool_name not in allowed:
            log_event("tool_call", user, request_id,
                      {"tool": tool_name, "status": "denied", "role": user_role})
            return {
                "success": False,
                "error": f"Access denied: role '{user_role}' is not permitted to use tool '{tool_name}'"
            }

    # ── Schema / parameter validation ─────────────────────────────────────
    schema_error = _validate_tool_schema(tool_name, kwargs)
    if schema_error:
        log_event("tool_call", user, request_id,
                  {"tool": tool_name, "status": "schema_error", "error": schema_error})
        return {"success": False, "error": schema_error}

    # ── Audit log before execution ─────────────────────────────────────────
    log_event("tool_call", user, request_id,
              {"tool": tool_name, "status": "executing", "arg_keys": list(kwargs.keys())})
    print(f"[AUDIT LOG] Tool executed: {tool_name} | Role: {user_role} | Args Keys: {list(kwargs.keys())}")

    tools = {
        "arima_forecast": arima_forecast,
        "anomaly_detection": anomaly_detection,
        "generate_graph": generate_graph,
        "export_csv": export_csv,
        "export_notes": export_notes,
    }

    if tool_name not in tools:
        log_event("tool_call", user, request_id,
                  {"tool": tool_name, "status": "unknown_tool"})
        return {"success": False, "error": f"Unknown tool: {tool_name}"}

    try:
        result = tools[tool_name](**kwargs)
        log_event("tool_call", user, request_id,
                  {"tool": tool_name, "status": "success", "result_keys": list(result.keys()) if isinstance(result, dict) else []})
        return result
    except Exception as exc:
        log_event("tool_call", user, request_id,
                  {"tool": tool_name, "status": "error", "error": str(exc)[:300]})
        return {"success": False, "error": f"Tool '{tool_name}' raised an exception: {exc}"}


if __name__ == "__main__":
    # Test tools
    print("Testing tools module...")
    
    # Test CSV export
    test_df = pd.DataFrame({
        "Month": ["Jan", "Feb", "Mar"],
        "Revenue": [10000, 12000, 11000]
    })
    result = export_csv(test_df, "test_export")
    print(f"CSV Export: {result}")
    
    # Test notes export
    result = export_notes("Test notes for Orion system", "test_notes")
    print(f"Notes Export: {result}")
    
    # Test anomaly detection
    test_series = pd.Series([10, 11, 12, 10, 11, 50, 12, 10])  # 50 is anomaly
    result = anomaly_detection(test_series)
    print(f"Anomaly Detection: {result}")
    
    print("\nTools module ready!")
