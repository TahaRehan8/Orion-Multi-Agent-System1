"""
Simulation Engine for Orion Multi-Agent RAG System
Allows testing agents with scenarios and analyzing results
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum

# Add parent path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.registry import get_agent, update_agent_prompt


class SimulationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TestCase:
    """A single test case for simulation"""
    id: str
    input: str
    expected_behavior: Optional[str] = None
    actual_output: Optional[str] = None
    passed: Optional[bool] = None
    error: Optional[str] = None
    duration_ms: int = 0


@dataclass
class SimulationRun:
    """A complete simulation run"""
    id: str
    agent_id: str
    created_by: str
    created_at: str
    status: SimulationStatus = SimulationStatus.PENDING
    test_cases: List[TestCase] = field(default_factory=list)
    passed_count: int = 0
    failed_count: int = 0
    analysis: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        result = asdict(self)
        result["status"] = self.status.value
        result["test_cases"] = [asdict(tc) for tc in self.test_cases]
        return result


# Storage for simulation runs
SIMULATION_PATH = Path(__file__).parent.parent / "outputs" / "simulations.json"
_simulation_runs: Dict[str, SimulationRun] = {}


def _load_simulations():
    """Load simulations from disk"""
    global _simulation_runs
    if SIMULATION_PATH.exists():
        try:
            with open(SIMULATION_PATH, "r") as f:
                data = json.load(f)
                for run_id, run_data in data.items():
                    run_data["status"] = SimulationStatus(run_data.get("status", "pending"))
                    run_data["test_cases"] = [TestCase(**tc) for tc in run_data.get("test_cases", [])]
                    _simulation_runs[run_id] = SimulationRun(**run_data)
        except Exception as e:
            print(f"[SIMULATION] Error loading simulations: {e}")


def _save_simulations():
    """Save simulations to disk"""
    try:
        SIMULATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {run_id: run.to_dict() for run_id, run in _simulation_runs.items()}
        with open(SIMULATION_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[SIMULATION] Error saving simulations: {e}")


_load_simulations()


# Rate limit mitigation: delay between test cases to avoid Mistral API rate limit exceeded
# Mistral free tier ~20 req/min; each test case triggers multiple API calls (orchestrator + agents)
SIMULATION_DELAY_BETWEEN_CASES_SEC = float(os.getenv("SIMULATION_DELAY_SEC", "3"))

# Always run exactly this many test cases for any agent (edge or smart) - reduces rate limit risk
MAX_SIMULATION_TEST_CASES = 2


# Edge case templates for different agent types
EDGE_CASE_TEMPLATES = {
    "hr": [
        "What is the salary of an employee who doesn't exist?",
        "Show me employees from a department called 'XYZ123'",
    ],
    "finance": [
        "What was the revenue in the year 1800?",
        "Show me transactions with negative amounts",
    ],
    "scheduler": [
        "What meetings are scheduled for February 30th?",
        "Show events from 1000 years ago",
    ],
    "default": [
        "Process this empty request:",
        "What happens if I ask about something completely unrelated?",
    ]
}


def generate_edge_cases(agent_id: str) -> List[TestCase]:
    """Generate edge case test scenarios for an agent"""
    agent = get_agent(agent_id)
    if not agent:
        return []
    
    # Determine which template to use
    template_key = "default"
    if agent_id in EDGE_CASE_TEMPLATES:
        template_key = agent_id
    elif "hr" in agent_id.lower() or "employee" in agent.description.lower():
        template_key = "hr"
    elif "finance" in agent_id.lower() or "budget" in agent.description.lower():
        template_key = "finance"
    elif "schedule" in agent_id.lower() or "calendar" in agent.description.lower():
        template_key = "scheduler"
    
    templates = EDGE_CASE_TEMPLATES[template_key][:MAX_SIMULATION_TEST_CASES]
    
    test_cases = []
    for i, template in enumerate(templates):
        test_cases.append(TestCase(
            id=f"edge_{i+1}",
            input=template,
            expected_behavior="Should handle gracefully without errors or hallucinations"
        ))
    
    return test_cases


def generate_smart_scenarios(agent_id: str) -> List[TestCase]:
    """Generate intelligent test scenarios based on agent's use case and allowed agents"""
    import httpx
    
    agent = get_agent(agent_id)
    if not agent:
        return generate_edge_cases(agent_id)
    
    # Get agent metadata
    metadata = agent.metadata or {}
    use_case = metadata.get('responses', {}).get('problem', agent.description)
    allowed_agents = metadata.get('allowed_agents', ['all'])
    
    # Build prompt for scenario generation
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    if not MISTRAL_API_KEY:
        return generate_edge_cases(agent_id)
    
    prompt = f"""Generate EXACTLY 2 test scenarios (no more, no less) for an AI agent with this configuration:

Agent Name: {agent.name}
Use Case: {use_case}
Allowed Sub-Agents: {', '.join(allowed_agents)}

Generate 2 scenarios that:
1. Test core functionality for the use case
2. Test edge cases (missing data, invalid inputs) or boundary between allowed and blocked functionality

Return ONLY JSON in this format: {{"scenarios": ["scenario 1", "scenario 2"]}}"""
    
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": prompt}]
    }
    
    max_retries = 3
    base_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = base_delay * (2 ** attempt)
                print(f"[SIMULATION] Mistral rate limit retry {attempt + 1}/{max_retries}, waiting {delay}s")
                time.sleep(delay)
            
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    json=payload, 
                    headers=headers
                )
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        continue  # Retry with delay on next iteration
                    print(f"[SIMULATION] Rate limit exceeded after {max_retries} retries")
                    return generate_edge_cases(agent_id)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
            
            # Parse JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            scenarios = data.get("scenarios", [])[:MAX_SIMULATION_TEST_CASES]
            
            test_cases = []
            for i, scenario in enumerate(scenarios):
                test_cases.append(TestCase(
                    id=f"smart_{i+1}",
                    input=scenario,
                    expected_behavior="Should respond correctly within allowed agent scope"
                ))
            
            return test_cases
            
        except Exception as e:
            print(f"[SIMULATION] Smart scenario generation failed: {e}")
            break
    
    return generate_edge_cases(agent_id)


def create_simulation(agent_id: str, created_by: str, test_cases: List[TestCase] = None) -> SimulationRun:
    """Create a new simulation run"""
    run_id = f"sim_{agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    run = SimulationRun(
        id=run_id,
        agent_id=agent_id,
        created_by=created_by,
        created_at=datetime.now().isoformat(),
        test_cases=test_cases or [],
        status=SimulationStatus.PENDING
    )
    
    _simulation_runs[run_id] = run
    _save_simulations()
    
    return run


def run_simulation(run_id: str) -> SimulationRun:
    """Execute a simulation run"""
    if run_id not in _simulation_runs:
        raise ValueError(f"Simulation {run_id} not found")
    
    run = _simulation_runs[run_id]
    run.status = SimulationStatus.RUNNING
    _save_simulations()
    
    # Import agent execution
    from backend.orchestrator import analyze_query_with_constraints, execute_task, synthesize_response
    from backend.registry import get_agent
    
    # Get agent config for constraints
    agent = get_agent(run.agent_id)
    allowed_agents = None
    if agent and agent.metadata:
        allowed_agents = agent.metadata.get('allowed_agents')
    
    passed = 0
    failed = 0
    
    for idx, test_case in enumerate(run.test_cases):
        try:
            # Rate limit mitigation: delay before each test case (except first) to space out Mistral API calls
            # Each test case triggers: orchestrator + sub-agents + possibly synthesis = multiple API calls
            if idx > 0 and SIMULATION_DELAY_BETWEEN_CASES_SEC > 0:
                print(f"[SIMULATION] Waiting {SIMULATION_DELAY_BETWEEN_CASES_SEC}s before test case {idx + 1} (rate limit mitigation)")
                time.sleep(SIMULATION_DELAY_BETWEEN_CASES_SEC)
            
            start_time = datetime.now()
            
            # Handle dict-formatted test cases (from smart scenarios)
            if isinstance(test_case.input, dict):
                query = test_case.input.get('input', {}).get('query', str(test_case.input))
                if isinstance(query, dict):
                    query = str(test_case.input)
            else:
                query = str(test_case.input)
            
            # Use constrained orchestrator for custom agents
            if allowed_agents and agent and agent.is_custom:
                tasks = analyze_query_with_constraints(query, allowed_agents)
            else:
                from backend.orchestrator import analyze_query
                tasks = analyze_query(query)
            
            # Execute tasks
            for task in tasks:
                task.result = execute_task(task, query)
            
            # Synthesize response
            result = synthesize_response(query, tasks)
            
            end_time = datetime.now()
            test_case.duration_ms = int((end_time - start_time).total_seconds() * 1000)
            test_case.actual_output = result[:2000] if len(result) > 2000 else result
            
            # Simple pass/fail based on whether we got a response without error keywords
            error_keywords = ["error", "exception", "failed", "cannot", "unable"]
            has_errors = any(kw in result.lower() for kw in error_keywords)
            
            test_case.passed = not has_errors
            if test_case.passed:
                passed += 1
            else:
                failed += 1
                
        except Exception as e:
            test_case.error = str(e)
            test_case.passed = False
            failed += 1
    
    run.passed_count = passed
    run.failed_count = failed
    run.status = SimulationStatus.COMPLETED
    
    # Generate analysis
    run.analysis = analyze_results(run)
    
    _save_simulations()
    return run


def run_simulation_streaming(run_id: str):
    """
    Execute simulation run as a generator, yielding progress updates for streaming UI.
    Yields progress strings, then the final SimulationRun.
    """
    if run_id not in _simulation_runs:
        raise ValueError(f"Simulation {run_id} not found")
    
    run = _simulation_runs[run_id]
    run.status = SimulationStatus.RUNNING
    _save_simulations()
    
    from backend.orchestrator import analyze_query_with_constraints, execute_task, synthesize_response
    from backend.registry import get_agent
    
    agent = get_agent(run.agent_id)
    allowed_agents = None
    if agent and agent.metadata:
        allowed_agents = agent.metadata.get('allowed_agents')
    
    passed = 0
    failed = 0
    total = len(run.test_cases)
    
    for idx, test_case in enumerate(run.test_cases):
        try:
            if idx > 0 and SIMULATION_DELAY_BETWEEN_CASES_SEC > 0:
                yield f"⏳ Waiting {int(SIMULATION_DELAY_BETWEEN_CASES_SEC)}s before next test (rate limit)...\n\n"
                time.sleep(SIMULATION_DELAY_BETWEEN_CASES_SEC)
            
            yield f"🔄 **Running test case {idx + 1}/{total}...**\n\n"
            
            start_time = datetime.now()
            
            if isinstance(test_case.input, dict):
                query = test_case.input.get('input', {}).get('query', str(test_case.input))
                if isinstance(query, dict):
                    query = str(test_case.input)
            else:
                query = str(test_case.input)
            
            if allowed_agents and agent and agent.is_custom:
                tasks = analyze_query_with_constraints(query, allowed_agents)
            else:
                from backend.orchestrator import analyze_query
                tasks = analyze_query(query)
            
            for task in tasks:
                task.result = execute_task(task, query)
            
            result = synthesize_response(query, tasks)
            
            end_time = datetime.now()
            test_case.duration_ms = int((end_time - start_time).total_seconds() * 1000)
            test_case.actual_output = result[:2000] if len(result) > 2000 else result
            
            error_keywords = ["error", "exception", "failed", "cannot", "unable"]
            has_errors = any(kw in result.lower() for kw in error_keywords)
            
            test_case.passed = not has_errors
            if test_case.passed:
                passed += 1
            else:
                failed += 1
            
            status_icon = "✅" if test_case.passed else "❌"
            yield f"{status_icon} **Test case {idx + 1}/{total} complete**\n\n"
                
        except Exception as e:
            test_case.error = str(e)
            test_case.passed = False
            failed += 1
            yield f"❌ **Test case {idx + 1}/{total} failed:** {str(e)[:100]}\n\n"
    
    run.passed_count = passed
    run.failed_count = failed
    run.status = SimulationStatus.COMPLETED
    run.analysis = analyze_results(run)
    _save_simulations()


def analyze_results(run: SimulationRun) -> str:
    """Analyze simulation results and provide recommendations"""
    total = len(run.test_cases)
    if total == 0:
        return "No test cases to analyze."
    
    pass_rate = (run.passed_count / total) * 100
    
    analysis = f"""**Simulation Analysis**

- **Pass Rate:** {pass_rate:.1f}% ({run.passed_count}/{total})
- **Failed Cases:** {run.failed_count}

"""
    
    if run.failed_count > 0:
        analysis += "**Failed Test Cases:**\n"
        for tc in run.test_cases:
            if not tc.passed:
                inp = str(tc.input)[:200] if len(str(tc.input)) > 200 else str(tc.input)
                analysis += f"- `{tc.id}`: {inp}\n"
                if tc.error:
                    analysis += f"  - Error: {tc.error}\n"
        
        # Add recommendations
        run.recommendations = [
            "Review failed test cases for common patterns",
            "Consider adding input validation to the agent prompt",
            "Test with more edge cases before deployment"
        ]
        
        analysis += "\n**Recommendations:**\n"
        for rec in run.recommendations:
            analysis += f"- {rec}\n"
    else:
        analysis += "All test cases passed. Agent appears robust."
        run.recommendations = ["Agent is ready for deployment"]
    
    return analysis


def get_simulation(run_id: str) -> Optional[SimulationRun]:
    """Get a simulation run by ID"""
    return _simulation_runs.get(run_id)


def get_simulations_for_agent(agent_id: str) -> List[SimulationRun]:
    """Get all simulations for an agent"""
    return [r for r in _simulation_runs.values() if r.agent_id == agent_id]


def modify_agent_from_simulation(run_id: str, modifications: str) -> bool:
    """Apply modifications to an agent based on simulation results"""
    run = get_simulation(run_id)
    if not run:
        return False
    
    agent = get_agent(run.agent_id)
    if not agent or not agent.is_custom:
        return False
    
    # Append modifications to the prompt
    new_prompt = f"{agent.prompt_template}\n\n---\nModifications based on simulation {run_id}:\n{modifications}"
    
    return update_agent_prompt(run.agent_id, new_prompt)


if __name__ == "__main__":
    # Test the simulation module
    print("Testing simulation module...")
    
    edge_cases = generate_edge_cases("finance")
    print(f"Generated {len(edge_cases)} edge cases for finance agent")
    
    for tc in edge_cases:
        print(f"  - {tc.input}")
