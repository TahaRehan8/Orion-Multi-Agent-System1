"""
Orchestrator Agent for Orion Multi-Agent RAG System
Coordinates HR, Finance, and Scheduler agents using Mistral Large for intelligent task breakdown
"""

import os
import json
import traceback

import httpx
from dotenv import load_dotenv
from typing import Generator, Optional
from dataclasses import dataclass, field
from enum import Enum

from langfuse import observe

from agents.model_tier import SMALL_TIER

load_dotenv()

# Security & audit imports (fail-safe: don't crash if modules unavailable)
try:
    from backend.security import preprocess_prompt, postprocess_output
    from backend.audit import log_event, generate_request_id
    _SECURITY_AVAILABLE = True
except ImportError:
    _SECURITY_AVAILABLE = False
    def preprocess_prompt(text, user=None): return {"approved": True, "reason": "security module unavailable", "clean_text": text}
    def postprocess_output(text): return text, []
    def log_event(*a, **kw): pass
    def generate_request_id(): return "ORION-NO-AUDIT"

# Mistral API configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Model name - using model tier system
ORCHESTRATOR_MODEL = SMALL_TIER


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    id: str
    description: str
    agent: str
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""


@dataclass
class OrchestrationResult:
    tasks: list = field(default_factory=list)
    final_response: str = ""
    agents_used: list = field(default_factory=list)
    success: bool = True


# Agent definitions for prompt building
AGENT_DEFINITIONS = {
    "hr": "hr: Handles employee data, departments, salaries, performance reviews, headcount",
    "finance": "finance: Handles financial data, budgets, expenses, revenue, transactions, costs, forecasting, anomaly detection",
    "scheduler": "scheduler: Handles calendar events, meetings, schedules, appointments, time-related queries",
    "chart": "chart: Generates graphs, charts, visualizations, plots data",
    "sql": "sql: Executes data queries, aggregations, filters data, runs SQL-like analysis"
}

ALL_AGENTS = ["hr", "finance", "scheduler", "chart", "sql"]

def _to_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False, default=str)
    except Exception:
        return str(x)


def _safe_lower(x) -> str:
    return _to_text(x).lower()


def build_task_breakdown_prompt(allowed_agents: list = None, custom_context: str = None) -> str:
    """Build orchestrator prompt with optional agent constraints"""
    
    # Only use built-in agents — custom agents are managed through the Lab Agent flow,
    # never routed to directly by the orchestrator
    all_known_agents = ALL_AGENTS.copy()
        
    # Build agent list for prompt
    agent_lines_list = []
    for a in all_known_agents:
        if a in AGENT_DEFINITIONS:
            agent_lines_list.append(f"- {AGENT_DEFINITIONS[a]}")
    agent_lines = "\n".join(agent_lines_list)
    
    constraint_note = ""
    if allowed_agents and set(allowed_agents) != set(ALL_AGENTS):
        str_enabled = ', '.join(allowed_agents)
        constraint_note = f"\n\nCURRENTLY ENABLED AGENTS: {str_enabled}\nIMPORTANT: Try to fulfill the query using ONLY the currently enabled agents. However, if a required task clearly belongs to the domain of a disabled agent (e.g., generating charts when the 'chart' agent is disabled, or querying db when 'sql' is disabled), you MUST still assign it to the disabled agent. Do NOT force a task into an inappropriate enabled agent (like forcing finance to draw a graph). The system will correctly intercept and block disabled agents."
    
    custom_section = ""
    if custom_context:
        custom_section = f"\n\nAGENT CONTEXT: {custom_context}"
    
    return f"""You are an orchestrator that analyzes user queries and breaks them down into tasks for specialized agents.

Available agents:
{agent_lines}{constraint_note}{custom_section}

Analyze the user query and determine:
1. Which agent(s) should handle this query
2. What specific task each agent should perform

IMPORTANT RULES:
- If the query only needs ONE agent, return a single task
- If the query needs MULTIPLE agents, break it into separate tasks (one per agent)
- Each task should have a clear, specific description of what that agent needs to do
- Keep task descriptions concise but actionable

Respond ONLY with a valid JSON object in this exact format:
{{
    "tasks": [
        {{"id": "task_1", "description": "Task description here", "agent": "agent_name"}}
    ],
    "reasoning": "Brief explanation of why these agents were chosen"
}}

Now analyze this query:
"""


# Default prompt for backward compatibility
TASK_BREAKDOWN_PROMPT = build_task_breakdown_prompt()


def generate_with_thinking(prompt: str, force_json: bool = False) -> str:
    """Generate content using Mistral Large"""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": ORCHESTRATOR_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    if force_json:
        payload["response_format"] = {"type": "json_object"}
    
    with httpx.Client(timeout=120.0) as client:
        response = client.post(MISTRAL_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


@observe(as_type="generation")
def analyze_query(query: str, allowed_agents: list = None) -> list[Task]:
    """
    Use Mistral Large to analyze the query and break it into tasks with agent assignments.
    """
    import time
    
    q_clean = query.strip().lower().rstrip("?.! ")
    greetings = ["hello", "hi", "hey", "greetings", "howdy", "good morning", "good afternoon", "good evening", "yo"]
    if q_clean in greetings:
        task = Task(
            id="task_1",
            description="Respond to greeting",
            agent="orchestrator",
            status=TaskStatus.PENDING
        )
        task.result = "Hello! I am Orion, your Multi-Agent RAG Assistant. How can I help you today?"
        return [task]

    print("\n" + "="*60)
    print("[ORCHESTRATOR] Analyzing Query")
    print("="*60)
    print(f"Query: {query}")
    print("-"*60)
    
    prompt = build_task_breakdown_prompt(allowed_agents) + f'"{ query}"'
    
    try:
        start_time = time.time()
        content = generate_with_thinking(prompt, force_json=True)
        elapsed = time.time() - start_time
        print(f"[ORCHESTRATOR] Query analysis completed in {elapsed:.2f}s")
        
        content = content.strip()
        
        # Try to parse JSON from response
        # Handle markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        data = _parse_json_robust(content)
        tasks = []
        
        print(f"[ORCHESTRATOR] Task breakdown:")
        for task_data in data.get("tasks", []):
            task_agent = task_data.get("agent", "finance")
            if allowed_agents and task_agent not in allowed_agents:
                print(f"  - BLOCKED: {task_agent} not in allowed agents")
                task = Task(
                    id=task_data.get("id", f"task_{len(tasks)+1}"),
                    description=f"BLOCKED: User requested a task for the '{task_agent.upper()}' agent, but it is currently disabled.",
                    agent="orchestrator",
                    status=TaskStatus.PENDING
                )
                task.result = f"Error: The {task_agent.upper()} agent is required for this task but is currently disabled."
                tasks.append(task)
                continue
            
            task = Task(
                id=task_data.get("id", f"task_{len(tasks)+1}"),
                description=task_data.get("description", ""),
                agent=task_agent,
                status=TaskStatus.PENDING
            )
            tasks.append(task)
            print(f"  - {task.id}: [{task.agent.upper()}] {task.description}")
        
        print(f"[ORCHESTRATOR] Reasoning: {data.get('reasoning', 'N/A')}")
        print("="*60 + "\n")
        
        if not tasks:
            if allowed_agents:
                return [Task(id="task_1", description="BLOCKED_DUE_TO_CONSTRAINTS", agent="orchestrator")]
            return [Task(id="task_1", description=query, agent="finance")]
            
        return tasks
        
    except (json.JSONDecodeError, Exception) as e:
        print(f"[ORCHESTRATOR] Task breakdown error: {e}")
        print(f"[ORCHESTRATOR] Using multi-keyword fallback")
        # Fallback: use keyword-based classification with multi-agent support
        fallback_agents = _keyword_fallback_multi(query)
        tasks = []
        for i, agent in enumerate(fallback_agents):
            tasks.append(Task(
                id=f"task_{i+1}",
                description=query,
                agent=agent
            ))
        print(f"[ORCHESTRATOR] Fallback assigned to: {[t.agent for t in tasks]}")
        print("="*60 + "\n")
        return tasks


def analyze_query_with_constraints(
    query: str,
    allowed_agents: list = None,
    custom_context: str = None
) -> list[Task]:
    """
    Analyze query with agent constraints for Lab Admin Agent.
    
    Args:
        query: User query to analyze
        allowed_agents: List of agent IDs that can be used (e.g., ['hr', 'finance'])
        custom_context: Optional context to inject (e.g., use case description)
    """
    import time
    
    q_clean = query.strip().lower().rstrip("?.! ")
    greetings = ["hello", "hi", "hey", "greetings", "howdy", "good morning", "good afternoon", "good evening", "yo"]
    if q_clean in greetings:
        task = Task(
            id="task_1",
            description="Respond to greeting",
            agent="orchestrator",
            status=TaskStatus.PENDING
        )
        task.result = "Hello! I am Orion, your Multi-Agent RAG Assistant. How can I help you today?"
        return [task]

    print("\n" + "="*60)
    print("[LAB ADMIN AGENT] Analyzing Query with Constraints")
    print("="*60)
    print(f"Query: {query}")
    print(f"Allowed Agents: {allowed_agents or 'ALL'}")
    print("-"*60)
    
    # Build constrained prompt
    prompt = build_task_breakdown_prompt(allowed_agents, custom_context) + f'"{query}"'
    
    try:
        start_time = time.time()
        content = generate_with_thinking(prompt)
        elapsed = time.time() - start_time
        print(f"[LAB ADMIN AGENT] Analysis completed in {elapsed:.2f}s")
        
        content = content.strip()
        
        # Parse JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        data = json.loads(content)
        tasks = []
        
        for task_data in data.get("tasks", []):
            agent = task_data.get("agent", "finance")
            
            # Reassign tasks to allowed agents if LLM picked a non-allowed one
            # (e.g. LLM routes to the custom agent's own ID instead of a sub-agent)
            if allowed_agents and agent not in allowed_agents:
                description = task_data.get("description", query)
                # Use keyword scoring to find the best allowed agent
                desc_lower = _safe_lower(description) + " " + _safe_lower(query)
                best_agent = allowed_agents[0]  # default fallback
                best_score = -1
                keyword_map = {
                    "hr": ["employee", "staff", "department", "salary", "performance", "hire", "team", "headcount"],
                    "finance": ["revenue", "expense", "budget", "transaction", "cost", "profit", "forecast", "anomaly", "financial"],
                    "scheduler": ["meeting", "calendar", "schedule", "event", "appointment"],
                    "chart": ["chart", "graph", "plot", "visualize", "visualization"],
                    "sql": ["query", "sql", "aggregate", "sum", "average", "count", "filter"],
                }
                for candidate in allowed_agents:
                    keywords = keyword_map.get(candidate, [])
                    score = sum(1 for kw in keywords if kw in desc_lower)
                    if score > best_score:
                        best_score = score
                        best_agent = candidate
                print(f"  - REASSIGNED: {agent} → {best_agent} (not in allowed agents)")
                agent = best_agent
                
            task = Task(
                id=task_data.get("id", f"task_{len(tasks)+1}"),
                description=task_data.get("description", ""),
                agent=agent,
                status=TaskStatus.PENDING
            )
            tasks.append(task)
            print(f"  - {task.id}: [{task.agent.upper()}] {task.description}")
        
        print(f"[LAB ADMIN AGENT] Tasks: {len(tasks)} valid")
        print("="*60 + "\n")
        
        if not tasks:
            # All tasks were filtered out
            return [Task(
                id="task_1",
                description=f"Cannot process this query with available agents ({', '.join(allowed_agents or ['none'])})",
                agent=allowed_agents[0] if allowed_agents else "finance"
            )]
        
        return tasks
        
    except Exception as e:
        print(f"[LAB ADMIN AGENT] Error: {e}")
        # Use first allowed agent as fallback
        fallback = (allowed_agents[0] if allowed_agents else "finance")
        return [Task(id="task_1", description=query, agent=fallback)]


def _parse_json_robust(content: str) -> dict:
    """Try multiple strategies to parse JSON from LLM output"""
    import re
    
    # Strategy 1: Direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract JSON object from surrounding text
    try:
        match = re.search(r'\{[\s\S]*\}', content)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Fix trailing commas
    try:
        cleaned = re.sub(r',\s*([}\]])', r'\1', content)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Fix single quotes to double quotes
    try:
        cleaned = content.replace("'", '"')
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # All strategies failed
    raise json.JSONDecodeError("All JSON parse strategies failed", content, 0)


def _keyword_fallback(query: str) -> str:
    """Simple keyword-based classification fallback (single agent)"""
    agents = _keyword_fallback_multi(query)
    return agents[0]


def _keyword_fallback_multi(query: str) -> list:
    """Keyword-based classification with multi-agent support for compound queries"""
    query_lower = query.lower()
    
    keyword_map = {
        "scheduler": ["meeting", "calendar", "schedule", "event", "appointment", 
                      "today", "tomorrow", "week", "monday", "tuesday", "wednesday",
                      "thursday", "friday", "saturday", "sunday", "date", "time"],
        "hr": ["employee", "staff", "department", "salary", "performance", 
               "hire", "hiring", "manager", "team", "worker", "personnel",
               "human resources", "hr", "job", "position", "headcount"],
        "finance": ["revenue", "expense", "budget", "transaction", "money",
                    "cost", "profit", "invoice", "payment", "financial",
                    "account", "balance", "income", "forecast", "anomaly"],
        "chart": ["chart", "graph", "plot", "visualize", "visualization", 
                  "bar chart", "line chart", "pie chart", "diagram"],
        "sql": ["query", "sql", "aggregate", "sum", "average", "count", 
                "group by", "filter", "data query"],
    }
    
    scores = {}
    for agent, keywords in keyword_map.items():
        scores[agent] = sum(1 for kw in keywords if kw in query_lower)
    
    max_score = max(scores.values())
    if max_score == 0:
        return ["finance"]
    
    # Detect compound queries: check for 'and' splitting different domains  
    # Return all agents that scored > 0 if query contains 'and'/'also'/'plus'
    compound_markers = [" and ", " also ", " plus ", " as well as "]
    is_compound = any(marker in query_lower for marker in compound_markers)
    
    if is_compound:
        matched = [agent for agent, score in scores.items() if score > 0]
        if len(matched) > 1:
            return matched
    
    # Single agent: return the highest scoring
    best = max(scores, key=scores.get)
    return [best]


@observe()
def execute_task(task: Task, query: str) -> str:
    """
    Execute a single task using the assigned agent.
    """
    import time
    
    # Import agents here to avoid circular imports
    from agents.hr_agent import ask_hr
    from agents.finance_agent import ask_finance
    from agents.scheduler_agent import ask_scheduler
    from agents.ChartAgent import ask_chart
    from agents.SQLAgent import ask_sql
    
    q_text = _to_text(query)
    desc_text = _to_text(task.description)

    # Build a specific prompt for the agent based on the task
    agent_prompt = f"{desc_text}\n\nOriginal user query for context: {q_text}"
    
    print(f"\n{'#'*60}")
    print(f"[ORCHESTRATOR] Executing Task: {task.id}")
    print(f"[ORCHESTRATOR] Agent: {task.agent.upper()}")
    print(f"[ORCHESTRATOR] Description: {desc_text}")
    print(f"{'#'*60}")
    
    start_time = time.time()
    
    try:
        if task.description.startswith("BLOCKED:"):
            elapsed = time.time() - start_time
            print(f"\n[ORCHESTRATOR] Task {task.id} BLOCKED")
            return getattr(task, 'result', f"Blocked: {task.agent} agent required.")
        elif task.agent == "orchestrator":
            return getattr(task, 'result', "Hello! How can I help you today?")
        elif task.agent == "hr":
            result = ask_hr(agent_prompt)
        elif task.agent == "finance":
            result = ask_finance(agent_prompt)
        elif task.agent == "scheduler":
            result = ask_scheduler(agent_prompt)
        elif task.agent == "chart":
            result = ask_chart(agent_prompt)
        elif task.agent == "sql":
            result = ask_sql(agent_prompt)
        elif task.agent.startswith("custom_"):
            from backend.registry import get_agent
            custom_agent = get_agent(task.agent)
            if custom_agent:
                allowed = custom_agent.metadata.get('allowed_agents', ['hr', 'finance', 'scheduler', 'chart', 'sql']) if custom_agent.metadata else None
                use_case = custom_agent.metadata.get('responses', {}).get('problem', '') if custom_agent.metadata else ''
                
                # Execute the constrained orchestrator logic
                sub_tasks = analyze_query_with_constraints(task.description, allowed, use_case)
                for st in sub_tasks:
                    # Prevent infinite recursion
                    if st.agent != task.agent:
                        st.result = execute_task(st, task.description)
                    else:
                        st.result = "Error: Custom agent self-reference"
                        
                result = synthesize_response(task.description, sub_tasks)
            else:
                result = f"Error: Custom agent {task.agent} not found."
        else:
            result = ask_finance(agent_prompt)
        
        elapsed = time.time() - start_time
        print(f"\n[ORCHESTRATOR] Task {task.id} completed in {elapsed:.2f}s")
        return _to_text(result)
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[ORCHESTRATOR] Task {task.id} FAILED after {elapsed:.2f}s")
        print(f"[ORCHESTRATOR] Error: {str(e)}")
        print("[ORCHESTRATOR] Traceback:")
        print(traceback.format_exc())
        raise


@observe(as_type="generation")
def synthesize_response(query: str, tasks: list) -> str:
    """
    Synthesize final response from all task results using Mistral.
    Runs post-processing (PII redaction) before returning.
    """
    if len(tasks) == 1:
        raw = tasks[0].result
    else:
        # Multiple tasks - synthesize results
        results_text = "\n\n".join([
            f"[{task.agent.upper()} Agent Result]:\n{task.result}"
            for task in tasks
        ])

        synthesis_prompt = f"""You are synthesizing results from multiple specialized agents to answer a user query.

User Query: {query}

Agent Results:
{results_text}

Provide a cohesive, well-organized response that combines insights from all agents. 
Format the response clearly with sections if multiple topics are covered.
Be concise but comprehensive.
IMPORTANT: If any Agent Result contains a Markdown image (e.g., ![Title](http...)), you MUST display it directly in your final response. Do NOT wrap the image link in backticks or code blocks. Just put the raw ![Title](url) on its own line."""

        try:
            raw = generate_with_thinking(synthesis_prompt)
        except Exception as e:
            raw = results_text

        # Bulletproof check: Ensure all markdown images from results survived synthesis
        import re
        images = re.findall(r'!\[.*?\]\(.*?\)', results_text)
        for img in images:
            # Mistral often wraps URLs in backticks. Strip them if they exist.
            raw = raw.replace(f"`{img}`", img)
            raw = raw.replace(f"```\n{img}\n```", img)
            if img not in raw:
                raw += f"\n\n{img}"

    # Post-process output (PII redaction)
    clean, flags = postprocess_output(raw)
    if flags:
        print(f"[SECURITY] Post-processing flags: {flags}")

    return clean


@observe()
def coordinate_stream(
    query: str,
    allowed_agents: list = None,
    user: str = "system",
    user_role: str = None,
) -> Generator[dict, None, None]:
    """
    Main coordination function that yields streaming updates.
    Includes: pre-processing (security), audit logging, request IDs, post-processing.

    Yields dictionaries with:
    - type: 'init', 'task_start', 'task_complete', 'synthesizing', 'final', 'error'
    """
    request_id = generate_request_id()

    # ── Step 0: Pre-processing (injection / abuse check) ──────────────────
    pre = preprocess_prompt(query, user=user)
    if not pre["approved"]:
        log_event("preprocess_block", user, request_id,
                  {"reason": pre["reason"], "raw_length": len(query)}, role=user_role)
        yield {"type": "error", "message": f"🚫 Request blocked: {pre['reason']}",
               "request_id": request_id}
        return

    clean_query = pre["clean_text"]
    log_event("preprocess_pass", user, request_id,
              {"clean_length": len(clean_query)}, role=user_role)

    # ── Step 1: Initial message ───────────────────────────────────────────
    log_event("init", user, request_id, {"allowed_agents": allowed_agents}, role=user_role)
    yield {"type": "init", "message": "Analyzing your query...", "request_id": request_id}

    # ── Step 2: Break down query into tasks ───────────────────────────────
    tasks = analyze_query(clean_query, allowed_agents)

    if len(tasks) == 1 and tasks[0].description == "BLOCKED_DUE_TO_CONSTRAINTS":
        log_event("error", user, request_id,
                  {"reason": "all tasks blocked by constraints"}, role=user_role)
        yield {"type": "error",
               "message": "Cannot process this query because the required agent is not selected. Please check your enabled agents.",
               "request_id": request_id}
        return

    # Fail-closed: if every task was blocked (BLOCKED: prefix), short-circuit
    real_tasks = [t for t in tasks if not t.description.startswith("BLOCKED:")]
    blocked_tasks = [t for t in tasks if t.description.startswith("BLOCKED:")]
    if blocked_tasks and not real_tasks:
        blocked_agents = [t.description for t in blocked_tasks]
        log_event("error", user, request_id,
                  {"reason": "fail-closed: all required agents disabled", "blocked": blocked_agents},
                  role=user_role)
        yield {"type": "error",
               "message": "🚫 All required agents for this request are disabled. " +
                          " | ".join(blocked_tasks[0].description for t in blocked_tasks[:3]),
               "request_id": request_id}
        return

    agents_used = list(set(task.agent for task in tasks if task.agent != "orchestrator"))

    log_event("tasks_identified", user, request_id,
              {"task_count": len(tasks), "agents": agents_used}, role=user_role)
    yield {
        "type": "tasks_identified",
        "tasks": [{"id": t.id, "description": t.description, "agent": t.agent, "status": t.status.value} for t in tasks],
        "agents": agents_used,
        "request_id": request_id,
    }

    # ── Step 3: Execute each task ─────────────────────────────────────────
    for i, task in enumerate(tasks):
        task.status = TaskStatus.IN_PROGRESS

        log_event("task_start", user, request_id,
                  {"task_id": task.id, "agent": task.agent, "description": task.description[:200]},
                  role=user_role)
        yield {
            "type": "task_start",
            "task_id": task.id,
            "task_index": i + 1,
            "total_tasks": len(tasks),
            "agent": task.agent,
            "description": task.description,
            "request_id": request_id,
        }

        try:
            result = execute_task(task, clean_query)
            task.result = result
            task.status = TaskStatus.COMPLETED
            log_event("task_complete", user, request_id,
                      {"task_id": task.id, "agent": task.agent, "status": "completed",
                       "result_length": len(result)}, role=user_role)
            yield {"type": "task_complete", "task_id": task.id, "agent": task.agent,
                   "result": result, "status": "completed", "request_id": request_id}
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.result = f"Error: {str(e)}"
            log_event("error", user, request_id,
                      {"task_id": task.id, "agent": task.agent, "error": str(e)[:500]},
                      role=user_role)
            yield {"type": "task_complete", "task_id": task.id, "agent": task.agent,
                   "result": task.result, "status": "failed", "request_id": request_id}

    # ── Step 4: Synthesize ────────────────────────────────────────────────
    if len(tasks) > 1:
        yield {"type": "synthesizing", "message": "Combining results from all agents...",
               "request_id": request_id}

    final_response = synthesize_response(clean_query, tasks)

    # Post-processing already applied inside synthesize_response, log any flags
    _, flags = postprocess_output(final_response)
    if flags:
        log_event("postprocess_redaction", user, request_id,
                  {"flags": flags}, role=user_role)

    log_event("final", user, request_id,
              {"agents_used": agents_used, "response_length": len(final_response),
               "success": all(t.status == TaskStatus.COMPLETED for t in tasks if not t.description.startswith("BLOCKED:"))},
              role=user_role)

    yield {
        "type": "final",
        "response": final_response,
        "tasks": [{"id": t.id, "description": t.description, "agent": t.agent,
                   "status": t.status.value, "result": t.result} for t in tasks],
        "agents_used": agents_used,
        "success": all(t.status == TaskStatus.COMPLETED for t in tasks if not t.description.startswith("BLOCKED:")),
        "request_id": request_id,
    }


def coordinate(query: str, allowed_agents: list = None) -> OrchestrationResult:
    """
    Non-streaming coordination - collects all results at once.
    Used for API responses that don't support streaming.
    """
    result = OrchestrationResult()
    
    for update in coordinate_stream(query, allowed_agents):
        if update["type"] == "final":
            result.final_response = update["response"]
            result.tasks = update["tasks"]
            result.agents_used = update["agents_used"]
            result.success = update["success"]
    
    return result
