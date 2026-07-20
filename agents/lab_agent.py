"""
Lab Agent for Orion Multi-Agent RAG System
Meta-agent responsible for agent creation, simulation, and management through guided conversation
"""

import os
import json
import time
import httpx
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

# Add parent path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.registry import register_custom_agent, get_all_agents, get_agent

load_dotenv()

# Mistral API configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


class LabMode(str, Enum):
    IDLE = "idle"
    CUSTOMIZE = "customize"
    DEPLOY = "deploy"
    SIMULATE = "simulate"
    CHAT = "chat"


class CustomizeStep(str, Enum):
    AGENT_NAME = "agent_name"
    PROBLEM = "problem"
    ALLOWED_AGENTS = "allowed_agents"
    FOCUS_AREAS = "focus_areas"
    CONSTRAINTS = "constraints"
    GENERATING = "generating"
    DONE = "done"


@dataclass
class LabSession:
    """State for a Lab Agent session"""
    mode: LabMode = LabMode.IDLE
    current_step: Optional[CustomizeStep] = None
    responses: Dict[str, str] = field(default_factory=dict)
    created_agent_id: Optional[str] = None
    user: Optional[str] = None
    
    def reset(self):
        self.mode = LabMode.IDLE
        self.current_step = None
        self.responses = {}
        self.created_agent_id = None


# Question prompts for each step
CUSTOMIZE_QUESTIONS = {
    CustomizeStep.AGENT_NAME: """**Step 1/5: Agent Name**

What would you like to name this agent?

_Enter a unique name for your custom agent (e.g., "HR Analyst", "Finance Monitor")._""",
    
    CustomizeStep.PROBLEM: """**Step 2/5: Use Case**

What problem should this agent solve?

_Describe the main use case. This will be injected into the agent's system prompt._""",
    
    CustomizeStep.ALLOWED_AGENTS: """**Step 3/5: Allowed Agents**

Which agents should this custom agent be able to use?

**Available agents:**
- `hr` - Employee data, salaries, departments, performance
- `finance` - Revenue, expenses, budgets, forecasting, anomalies
- `scheduler` - Meetings, calendar events, appointments
- `chart` - Graphs and visualizations
- `sql` - Data queries and aggregations

_Enter agent names separated by commas (e.g., "hr, finance") or 'all' for all agents._""",
    
    CustomizeStep.FOCUS_AREAS: """**Step 4/5: Focus Areas**

What should the agent focus on?

- Anomaly detection
- Summarization
- Forecasting
- Data analysis
- Report generation
- Q&A

_Enter focus areas or 'default' for general analysis._""",
    
    CustomizeStep.CONSTRAINTS: """**Step 5/5: Constraints**

Any specific constraints or requirements?

_For example: response length limits, output formats, etc._
_Type 'none' if there are no constraints._"""
}


def call_mistral(prompt: str) -> str:
    """Call Mistral API"""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": prompt}]
    }
    
    with httpx.Client(timeout=60.0) as client:
        response = client.post(MISTRAL_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


def generate_agent_prompt(responses: Dict[str, str]) -> str:
    """Generate a dynamic agent prompt based on user responses"""
    
    prompt = f"""Based on the following requirements, create a system prompt for a specialized AI agent:

Problem to solve: {responses.get('problem', 'General assistance')}

Data types to review: {responses.get('review_types', 'Text')}

Focus areas: {responses.get('focus_areas', 'General analysis')}

Constraints: {responses.get('constraints', 'None specified')}

Tool requirements: {responses.get('tool_requirements', 'Default tools')}

Compliance/Safety: {responses.get('compliance', 'Standard safety guidelines')}

Additional requirements: {responses.get('other_requirements', 'None')}

Generate a clear, professional system prompt for this agent. The prompt should:
1. Define the agent's role and expertise
2. Specify how it should approach problems
3. List what data types it can work with
4. Include any constraints or guidelines
5. Be concise but comprehensive

Return ONLY the system prompt text, no explanations."""

    try:
        generated_prompt = call_mistral(prompt)
        return generated_prompt.strip()
    except Exception as e:
        # Fallback prompt generation
        return f"""You are a specialized AI assistant focused on: {responses.get('problem', 'General assistance')}.

Your capabilities include analyzing: {responses.get('review_types', 'text documents')}.

Your primary focus areas are: {responses.get('focus_areas', 'data analysis and insights')}.

Guidelines:
- {responses.get('constraints', 'Provide accurate and helpful responses')}
- {responses.get('compliance', 'Follow standard safety and privacy guidelines')}

{responses.get('other_requirements', '')}

Always provide clear, actionable insights based on the data provided."""


def determine_tools(responses: Dict[str, str]) -> List[str]:
    """Determine which tools the agent should have based on responses"""
    tools = []
    
    tool_req = responses.get('tool_requirements', 'default').lower()
    focus = responses.get('focus_areas', '').lower()
    review_types = responses.get('review_types', '').lower()
    
    # Default tools
    if 'default' in tool_req or tool_req == '':
        tools = ["search_documents", "export_notes"]
    
    # Add based on focus areas
    if 'forecast' in focus or 'forecasting' in focus:
        tools.append("arima_forecast")
    
    if 'anomaly' in focus or 'outlier' in focus:
        tools.append("anomaly_detection")
    
    if 'graph' in focus or 'visual' in focus or 'chart' in focus:
        tools.append("generate_graph")
    
    if 'report' in focus or 'export' in focus:
        tools.extend(["export_csv", "export_notes"])
    
    # Add based on review types
    if 'excel' in review_types or 'tabular' in review_types or 'data' in review_types:
        tools.extend(["export_csv", "generate_graph"])
    
    if 'graph' in review_types or 'visual' in review_types:
        tools.append("generate_graph")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tools = []
    for tool in tools:
        if tool not in seen:
            seen.add(tool)
            unique_tools.append(tool)
    
    return unique_tools if unique_tools else ["search_documents", "export_notes"]


class LabAgent:
    """Lab Agent for managing agent creation and simulation"""
    
    def __init__(self):
        self.sessions: Dict[str, LabSession] = {}
    
    def get_session(self, user: str) -> LabSession:
        """Get or create a session for a user"""
        if user not in self.sessions:
            self.sessions[user] = LabSession(user=user)
        return self.sessions[user]
    
    def start_customize(self, user: str) -> str:
        """Start the customize workflow"""
        session = self.get_session(user)
        session.reset()
        session.mode = LabMode.CUSTOMIZE
        session.current_step = CustomizeStep.AGENT_NAME
        
        return f"""**Welcome to Agent Builder**

I'll guide you through creating a custom agent in 5 steps.

---

{CUSTOMIZE_QUESTIONS[CustomizeStep.AGENT_NAME]}"""
    
    def process_customize_response(self, user: str, response: str) -> str:
        """Process a response in the customize workflow"""
        session = self.get_session(user)
        
        if session.mode != LabMode.CUSTOMIZE:
            return "No active customization session. Type 'customize' to start."
        
        # Store the response
        session.responses[session.current_step.value] = response
        
        # Move to next step
        steps = list(CustomizeStep)
        current_idx = steps.index(session.current_step)
        next_idx = current_idx + 1
        
        # Check if next step is a question step (not GENERATING or DONE)
        if next_idx < len(steps) and steps[next_idx] in CUSTOMIZE_QUESTIONS:
            session.current_step = steps[next_idx]
            return CUSTOMIZE_QUESTIONS[session.current_step]
        else:
            # Generate the agent (we've answered all questions)
            session.current_step = CustomizeStep.GENERATING
            return self._generate_custom_agent(user)
    
    def _generate_custom_agent(self, user: str) -> str:
        """Generate and register the custom agent"""
        session = self.get_session(user)
        
        try:
            # Get agent name from user input
            name = session.responses.get('agent_name', 'Custom Agent').strip()
            
            # Get use case and build prompt
            use_case = session.responses.get('problem', 'General assistance')
            focus = session.responses.get('focus_areas', 'general analysis')
            constraints = session.responses.get('constraints', 'none')
            
            # Parse allowed agents
            allowed_agents_str = session.responses.get('allowed_agents', 'all').lower()
            if allowed_agents_str == 'all':
                allowed_agents = ['hr', 'finance', 'scheduler', 'chart', 'sql']
            else:
                allowed_agents = [a.strip() for a in allowed_agents_str.split(',')]
            
            # Build constraint prompt
            prompt_template = f"""You are a specialized AI assistant named "{name}".

USE CASE: {use_case}

FOCUS AREAS: {focus}

ALLOWED AGENTS: You can ONLY route queries to these agents: {', '.join(allowed_agents)}
Do NOT use any other agents. If a query requires an agent not in this list, politely decline.

CONSTRAINTS: {constraints}

Always provide clear, actionable insights based on the data provided."""
            
            # Determine tools based on allowed agents
            tools = ["search_documents", "export_notes"]
            if 'chart' in allowed_agents:
                tools.append("generate_graph")
            if 'finance' in allowed_agents:
                tools.extend(["arima_forecast", "anomaly_detection"])
            if any(a in allowed_agents for a in ['hr', 'finance']):
                tools.append("export_csv")
            
            # Remove duplicates
            tools = list(dict.fromkeys(tools))
            
            # Generate description
            description = f"{name}: {use_case[:100]}"
            
            # Register the agent
            agent = register_custom_agent(
                name=name,
                description=description,
                tools=tools,
                prompt_template=prompt_template,
                created_by=user,
                metadata={
                    "responses": session.responses,
                    "allowed_agents": allowed_agents,
                    "created_via": "lab_admin_agent"
                }
            )
            
            session.created_agent_id = agent.id
            session.current_step = CustomizeStep.DONE
            session.mode = LabMode.IDLE
            
            return f"""**Agent Created Successfully!**

---

**Agent Details:**
- **ID:** `{agent.id}`
- **Name:** {agent.name}
- **Status:** Draft
- **Allowed Agents:** {', '.join(allowed_agents)}
- **Tools:** {', '.join(tools)}

---

**Commands:**
- `list` - View all your agents
- `deploy` - Deploy this agent
- `test {agent.id}` - Run simulation tests
- `delete {agent.id}` - Delete this agent
"""
            
        except Exception as e:
            session.mode = LabMode.IDLE
            return f"""**Error Creating Agent**

Something went wrong: {str(e)}

Please type 'customize' to try again."""
    
    def start_deploy(self, user: str) -> str:
        """Start the deploy workflow"""
        session = self.get_session(user)
        session.mode = LabMode.DEPLOY
        
        if session.created_agent_id:
            return f"""**Deploy Agent**

Ready to deploy agent: `{session.created_agent_id}`

Type **'yes'** to deploy or **'no'** to cancel."""
        else:
            # List available custom agents
            agents = [a for a in get_all_agents() if a.is_custom]
            if not agents:
                session.mode = LabMode.IDLE
                return "No custom agents available to deploy. Type 'customize' to create one."
            
            agent_list = "\n".join([f"- `{a.id}` - {a.name} (Status: {a.status.value})" for a in agents])
            return f"""**Deploy Agent**

Available custom agents:
{agent_list}

Type the agent ID to deploy, or 'cancel' to abort."""
    
    def process_deploy_response(self, user: str, response: str) -> str:
        """Process a deploy response"""
        session = self.get_session(user)
        response_lower = response.strip().lower()
        
        if response_lower in ['no', 'cancel']:
            session.mode = LabMode.IDLE
            return "Deployment cancelled."
        
        from backend.registry import deploy_agent
        
        if response_lower == 'yes' and session.created_agent_id:
            agent_id = session.created_agent_id
        else:
            agent_id = response.strip()
        
        if deploy_agent(agent_id):
            session.mode = LabMode.IDLE
            return f"""**Agent Deployed!**

Agent `{agent_id}` is now active and ready to use.

You can now use this agent through the main chat interface."""
        else:
            return f"Could not deploy agent `{agent_id}`. Check if the ID is correct."
    
    def start_simulate(self, user: str) -> str:
        """Start the simulation workflow"""
        session = self.get_session(user)
        session.mode = LabMode.SIMULATE
        
        agents = [a for a in get_all_agents() if a.is_custom or a.id in ['hr', 'finance', 'scheduler']]
        agent_list = "\n".join([f"- `{a.id}` - {a.name}" for a in agents])
        
        return f"""**Simulate Scenarios**

Select an agent to test:
{agent_list}

Type the agent ID to begin simulation."""
    
    def process_simulate_response(self, user: str, response: str) -> str:
        """Process a simulate response"""
        session = self.get_session(user)
        response_lower = response.strip().lower()
        
        if response_lower == 'cancel':
            session.mode = LabMode.IDLE
            return "Simulation cancelled."
        
        # Check if agent is already selected (from test command or previous step)
        if 'simulate_agent' in session.responses:
            agent_id = session.responses['simulate_agent']
            agent = get_agent(agent_id)
            
            if not agent:
                session.mode = LabMode.IDLE
                return f"Agent `{agent_id}` not found. Session reset."
            
            # Handle simulation type selection
            if response_lower in ['1', 'edge', 'edge cases']:
                from backend.simulation import generate_edge_cases, create_simulation, run_simulation
                
                test_cases = generate_edge_cases(agent_id)
                sim = create_simulation(agent_id, user, test_cases)
                
                # Actually run the simulation
                result = run_simulation(sim.id)
                
                session.mode = LabMode.IDLE
                
                # Format results - show full scenario names and output (no truncation mid-line)
                results_text = []
                for tc in result.test_cases[:5]:
                    status = "✅" if tc.passed else "❌"
                    if isinstance(tc.input, dict):
                        input_text = tc.input.get('input', {}).get('query', str(tc.input))
                        if isinstance(input_text, dict):
                            input_text = str(tc.input)
                    else:
                        input_text = str(tc.input)
                    output = tc.actual_output or "No output"
                    results_text.append(f"{status} **{input_text}**\n   →\n{output}\n")
                
                return f"""**Edge Case Test Complete**

Tested {len(test_cases)} scenarios for **{agent.name}**

**Results:** {result.passed_count} passed, {result.failed_count} failed

{chr(10).join(results_text)}

{result.analysis if result.analysis else ''}"""
            
            elif response_lower in ['2', 'smart', 'auto']:
                from backend.simulation import generate_smart_scenarios, create_simulation, run_simulation
                
                test_cases = generate_smart_scenarios(agent_id)
                # Rate limit mitigation: brief pause after Mistral call for scenario generation
                # before run_simulation triggers more API calls (orchestrator + agents per test case)
                time.sleep(2)
                sim = create_simulation(agent_id, user, test_cases)
                
                # Actually run the simulation
                result = run_simulation(sim.id)
                
                session.mode = LabMode.IDLE
                
                # Format results - show full scenario names and output (no truncation mid-line)
                results_text = []
                for tc in result.test_cases[:5]:
                    status = "✅" if tc.passed else "❌"
                    if isinstance(tc.input, dict):
                        input_text = tc.input.get('input', {}).get('query', str(tc.input))
                        if isinstance(input_text, dict):
                            input_text = str(tc.input)
                    else:
                        input_text = str(tc.input)
                    output = tc.actual_output or "No output"
                    results_text.append(f"{status} **{input_text}**\n   →\n{output}\n")
                
                return f"""**Smart Scenario Test Complete**

Tested {len(test_cases)} AI-generated scenarios for **{agent.name}**

**Results:** {result.passed_count} passed, {result.failed_count} failed

{chr(10).join(results_text)}

{result.analysis if result.analysis else ''}"""
            
            elif response_lower in ['3', 'custom', 'user']:
                session.mode = LabMode.IDLE
                return f"""**Custom Test Mode**

Enter a test query for agent **{agent.name}**.
Type `use {agent_id}` to start chatting with this agent."""
            
            else:
                return "Please type **1**, **2**, or **3** to select simulation type."
        
        # No agent selected yet - expect agent ID
        agent = get_agent(response.strip())
        if not agent:
            return f"Agent `{response.strip()}` not found. Please enter a valid agent ID."
        
        # Store agent for simulation
        session.responses['simulate_agent'] = response.strip()
        
        return f"""**Testing Agent: {agent.name}**

Select simulation type:
1. **User scenarios** - Provide your own test cases
2. **Edge cases** - Auto-generate challenging scenarios  
3. **Smart test** - AI-generated context-aware scenarios

Type 1, 2, or 3 to continue."""
    
    def process_message(self, user: str, message: str) -> str:
        """Main entry point - process any message to Lab Agent"""
        session = self.get_session(user)
        message_lower = message.strip().lower()
        
        # Check for command triggers
        if message_lower == 'customize':
            return self.start_customize(user)
        elif message_lower == 'deploy':
            return self.start_deploy(user)
        elif message_lower == 'simulate' or message_lower == 'simulate scenarios':
            return self.start_simulate(user)
        elif message_lower == 'list':
            return self._list_agents(user)
        elif message_lower.startswith('delete '):
            agent_id = message.strip()[7:].strip()
            return self._delete_agent(user, agent_id)
        elif message_lower.startswith('test '):
            agent_id = message.strip()[5:].strip()
            return self._test_agent(user, agent_id)
        elif message_lower.startswith('use '):
            agent_id = message.strip()[4:].strip()
            return self._use_agent(user, agent_id)
        elif message_lower == 'cancel' or message_lower == 'reset':
            session.reset()
            return "Session reset. Available commands: **customize**, **deploy**, **use <id>**, **test <id>**"
        elif message_lower == 'help':
            return self._get_help()
        
        # Process based on current mode
        if session.mode == LabMode.CUSTOMIZE:
            return self.process_customize_response(user, message)
        elif session.mode == LabMode.DEPLOY:
            return self.process_deploy_response(user, message)
        elif session.mode == LabMode.SIMULATE:
            return self.process_simulate_response(user, message)
        elif session.mode == LabMode.CHAT:
            return self._chat_with_agent(user, message)
        else:
            return self._get_welcome()
    
    def _run_simulation_streaming(self, user: str, response_lower: str, agent_id: str, agent) -> None:
        """Generator that yields progress during simulation (edge or smart)."""
        session = self.get_session(user)
        
        if response_lower in ['1', 'edge', 'edge cases']:
            from backend.simulation import generate_edge_cases, create_simulation, run_simulation_streaming, get_simulation
            
            yield "🔄 **Generating edge case scenarios...**\n\n"
            test_cases = generate_edge_cases(agent_id)
            sim = create_simulation(agent_id, user, test_cases)
            
            yield "🔄 **Running simulation...**\n\n"
            for chunk in run_simulation_streaming(sim.id):
                yield chunk
            
            result = get_simulation(sim.id)
            session.mode = LabMode.IDLE
            
            results_text = []
            for tc in result.test_cases[:5]:
                status = "✅" if tc.passed else "❌"
                if isinstance(tc.input, dict):
                    input_text = tc.input.get('input', {}).get('query', str(tc.input))
                    if isinstance(input_text, dict):
                        input_text = str(tc.input)
                else:
                    input_text = str(tc.input)
                output = tc.actual_output or "No output"
                results_text.append(f"{status} **{input_text}**\n   →\n{output}\n")
            
            yield f"""**Edge Case Test Complete**

Tested {len(test_cases)} scenarios for **{agent.name}**

**Results:** {result.passed_count} passed, {result.failed_count} failed

{chr(10).join(results_text)}

{result.analysis if result.analysis else ''}"""
        
        elif response_lower in ['2', 'smart', 'auto']:
            from backend.simulation import generate_smart_scenarios, create_simulation, run_simulation_streaming, get_simulation
            
            yield "🔄 **Generating AI scenarios...**\n\n"
            test_cases = generate_smart_scenarios(agent_id)
            time.sleep(2)
            sim = create_simulation(agent_id, user, test_cases)
            
            yield "🔄 **Running simulation...**\n\n"
            for chunk in run_simulation_streaming(sim.id):
                yield chunk
            
            result = get_simulation(sim.id)
            session.mode = LabMode.IDLE
            
            results_text = []
            for tc in result.test_cases[:5]:
                status = "✅" if tc.passed else "❌"
                if isinstance(tc.input, dict):
                    input_text = tc.input.get('input', {}).get('query', str(tc.input))
                    if isinstance(input_text, dict):
                        input_text = str(tc.input)
                else:
                    input_text = str(tc.input)
                output = tc.actual_output or "No output"
                results_text.append(f"{status} **{input_text}**\n   →\n{output}\n")
            
            yield f"""**Smart Scenario Test Complete**

Tested {len(test_cases)} AI-generated scenarios for **{agent.name}**

**Results:** {result.passed_count} passed, {result.failed_count} failed

{chr(10).join(results_text)}

{result.analysis if result.analysis else ''}"""
    
    def process_message_streaming(self, user: str, message: str):
        """Generator that yields chunks for streaming UI. Use for simulate edge/smart."""
        session = self.get_session(user)
        message_lower = message.strip().lower()
        
        # Check if we're in simulate flow and user selected edge (1) or smart (2)
        if session.mode == LabMode.SIMULATE and 'simulate_agent' in session.responses:
            agent_id = session.responses['simulate_agent']
            agent = get_agent(agent_id)
            
            if agent and message_lower in ['1', '2', 'edge', 'edge cases', 'smart', 'auto']:
                for chunk in self._run_simulation_streaming(user, message_lower, agent_id, agent):
                    yield chunk
                return
        
        # Non-streaming path: get full response and yield once (with typing effect handled by frontend)
        response = self.process_message(user, message)
        yield response
    
    def _list_agents(self, user: str) -> str:
        """List all custom agents"""
        from backend.registry import get_custom_agents
        agents = get_custom_agents()
        
        if not agents:
            return "No custom agents yet. Type `customize` to create one."
        
        lines = ["**Your Custom Agents:**\n"]
        for a in agents:
            allowed = a.metadata.get('allowed_agents', ['all']) if a.metadata else ['all']
            lines.append(f"- `{a.id}` - **{a.name}** ({a.status.value})")
            lines.append(f"  - Allowed: {', '.join(allowed)}")
        
        lines.append("\n**Commands:** `deploy`, `test <id>`, `delete <id>`")
        return "\n".join(lines)
    
    def _delete_agent(self, user: str, agent_id: str) -> str:
        """Delete a custom agent"""
        from backend.registry import delete_custom_agent, get_agent
        
        agent = get_agent(agent_id)
        if not agent:
            return f"Agent `{agent_id}` not found."
        
        if not agent.is_custom:
            return f"Cannot delete built-in agent `{agent_id}`."
        
        if delete_custom_agent(agent_id):
            return f"**Deleted:** Agent `{agent_id}` ({agent.name}) has been removed."
        else:
            return f"Failed to delete agent `{agent_id}`."
    
    def _test_agent(self, user: str, agent_id: str) -> str:
        """Start simulation test for an agent"""
        from backend.registry import get_agent
        
        agent = get_agent(agent_id)
        if not agent:
            return f"Agent `{agent_id}` not found. Type `list` to see available agents."
        
        session = self.get_session(user)
        session.mode = LabMode.SIMULATE
        session.responses['simulate_agent'] = agent_id
        
        return f"""**Testing Agent: {agent.name}**

Select simulation type:
1. **edge** - Test with edge cases
2. **smart** - AI-generated scenarios
3. **custom** - Enter your own test queries

Type 1, 2, or 3 to continue."""
    
    def _use_agent(self, user: str, agent_id: str) -> str:
        """Start chatting with a custom agent"""
        from backend.registry import get_agent
        
        agent = get_agent(agent_id)
        if not agent:
            return f"Agent `{agent_id}` not found. Type `list` to see available agents."
        
        session = self.get_session(user)
        session.mode = LabMode.CHAT
        session.responses['chat_agent'] = agent_id
        
        allowed = agent.metadata.get('allowed_agents', ['all']) if agent.metadata else ['all']
        
        return f"""**Now using: {agent.name}**

Agent constraints:
- Allowed agents: {', '.join(allowed)}

You can now ask questions. Type `cancel` to stop using this agent."""
    
    def _chat_with_agent(self, user: str, message: str) -> str:
        """Process a chat message with the active custom agent"""
        session = self.get_session(user)
        agent_id = session.responses.get('chat_agent')
        
        if not agent_id:
            session.mode = LabMode.IDLE
            return "No agent selected. Use `use <agent_id>` to start."
        
        from backend.registry import get_agent
        from backend.orchestrator import analyze_query_with_constraints, execute_task, synthesize_response
        
        agent = get_agent(agent_id)
        if not agent:
            session.mode = LabMode.IDLE
            return f"Agent `{agent_id}` no longer exists."
        
        # Get allowed agents from metadata
        allowed_agents = agent.metadata.get('allowed_agents', ['hr', 'finance', 'scheduler', 'chart', 'sql']) if agent.metadata else None
        use_case = agent.metadata.get('responses', {}).get('problem', '') if agent.metadata else ''
        
        try:
            # Route through constrained orchestrator
            tasks = analyze_query_with_constraints(message, allowed_agents, use_case)
            
            # Execute each task
            for task in tasks:
                result = execute_task(task, message)
                task.result = result
            
            # Synthesize response
            response = synthesize_response(message, tasks)
            
            return f"""**{agent.name} Response:**

{response}

---
_Ask another question or type `cancel` to exit._"""
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _get_welcome(self) -> str:
        return """**Welcome to Orion Lab**

Create, manage, and chat with custom agents.

**Commands:**
- **customize** - Create a new agent
- **list** - View your agents
- **use <id>** - Chat with a custom agent
- **test <id>** - Run simulation tests
- **deploy** - Deploy an agent
- **delete <id>** - Remove an agent

What would you like to do?"""
    
    def _get_help(self) -> str:
        return """**Lab Agent Commands**

| Command | Description |
|---------|-------------|
| `customize` | Create a new custom agent |
| `list` | View all your agents |
| `use <id>` | Chat with a custom agent |
| `test <id>` | Run simulation tests |
| `deploy` | Deploy to production |
| `delete <id>` | Delete agent |
| `cancel` | Cancel current workflow |

**Tips:**
- Type `list` to see agent IDs
- Use `use <id>` to chat with an agent"""


# Singleton instance
lab_agent = LabAgent()


def ask_lab(question: str, user: str = "anonymous") -> str:
    """Main entry point for Lab Agent queries"""
    return lab_agent.process_message(user, question)


def ask_lab_stream(question: str, user: str = "anonymous"):
    """Generator that yields Lab Agent response chunks for streaming UI with typing animation."""
    for chunk in lab_agent.process_message_streaming(user, question):
        yield chunk


if __name__ == "__main__":
    # Test the lab agent
    print(ask_lab("help", "test_user"))
    print("\n" + "="*60 + "\n")
    print(ask_lab("customize", "test_user"))
