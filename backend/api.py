"""
FastAPI Backend for Orion Multi-Agent RAG System
Provides REST API endpoints with orchestrator-based multi-agent coordination
"""

import sys
sys.path.insert(0, '.')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import json
from io import BytesIO
from gtts import gTTS

# Import orchestrator
from backend.orchestrator import coordinate, coordinate_stream

# Security pre-processing (fail-safe)
try:
    from backend.security import preprocess_prompt
except ImportError:
    def preprocess_prompt(text, user=None): return {"approved": True, "reason": "unavailable", "clean_text": text}

# Initialize FastAPI app
app = FastAPI(
    title="Orion Multi-Agent RAG API",
    description="API with intelligent orchestrator for multi-agent coordination",
    version="3.0.0"
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for graph images (must be BEFORE route definitions)
# We define paths here and mount after OUTPUT_PATH is defined


# ============ Pydantic Models ============

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str
    stream: bool = False
    allowed_agents: Optional[List[str]] = None


class TaskInfo(BaseModel):
    """Model for task information"""
    id: str
    description: str
    agent: str
    status: str
    result: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str
    tasks: List[TaskInfo]
    agents_used: List[str]
    success: bool


class AgentInfo(BaseModel):
    """Model for agent information"""
    id: str
    name: str
    description: str


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    message: str


class AuthRequest(BaseModel):
    username: str
    password: str

class AuthResponse(BaseModel):
    success: bool
    message: str
    username: Optional[str] = None
    role: Optional[str] = None
    token: Optional[str] = None

class CreateAgentRequest(BaseModel):
    name: str
    description: str
    tools: List[str]
    prompt_template: str
    created_by: str
    metadata: Optional[dict] = None

class TTSRequest(BaseModel):
    text: str

# ============ Endpoints ============

@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Orion Multi-Agent RAG API",
        "version": "3.0.0",
        "docs": "/docs",
        "endpoints": ["/health", "/agents", "/chat", "/chat/stream"]
    }

@app.post("/tts", tags=["System"])
async def text_to_speech(req: TTSRequest):
    """Generate audio from text using gTTS and return as a stream"""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    try:
        tts = gTTS(req.text, lang='en')
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return StreamingResponse(fp, media_type="audio/mpeg")
    except Exception as e:
        print(f"[TTS ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint to verify API is running"""
    return HealthResponse(
        status="healthy",
        message="Orion Multi-Agent RAG API is running"
    )


@app.get("/agents", response_model=list[AgentInfo], tags=["Agents"])
async def get_agents():
    """Get list of available agents with their descriptions"""
    agents = [
        AgentInfo(
            id="orchestrator",
            name="Orchestrator",
            description="Coordinates all agents and breaks down complex queries"
        ),
        AgentInfo(
            id="scheduler",
            name="Scheduler Agent",
            description="Handles calendar events, meetings, and scheduling queries"
        ),
        AgentInfo(
            id="hr",
            name="HR Agent",
            description="Assists with employee data, departments, and performance information"
        ),
        AgentInfo(
            id="finance",
            name="Finance Agent",
            description="Provides financial data analysis, metrics, budget insights, forecasting and anomaly detection"
        ),
        AgentInfo(
            id="chart",
            name="Chart Agent",
            description="Generates graphs, charts, and data visualizations"
        ),
        AgentInfo(
            id="sql",
            name="SQL Agent",
            description="Executes data queries and aggregations on available data sources"
        )
    ]
    
    # Add custom deployed agents
    try:
        from backend.registry import get_custom_agents, AgentStatus
        custom_agents = get_custom_agents()
        for a in custom_agents:
            if a.status == AgentStatus.DEPLOYED:
                agents.append(
                    AgentInfo(
                        id=a.id,
                        name=f"{a.name} (Custom)",
                        description=a.description or "Custom Lab Agent"
                    )
                )
    except Exception as e:
        print(f"Error loading custom agents: {e}")
        
    return agents

@app.get("/agents/all", tags=["Agents"])
async def get_all_detailed_agents():
    """Get all agents including drafts with full details"""
    from backend.registry import get_all_agents
    return [a.to_dict() for a in get_all_agents()]

@app.post("/agents/custom", tags=["Agents"])
async def create_custom_agent(request: CreateAgentRequest):
    """Create a new custom agent"""
    try:
        from backend.registry import register_custom_agent
        agent = register_custom_agent(
            name=request.name,
            description=request.description,
            tools=request.tools,
            prompt_template=request.prompt_template,
            created_by=request.created_by,
            metadata=request.metadata or {}
        )
        return {"success": True, "agent": agent.to_dict()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/agents/{agent_id}/deploy", tags=["Agents"])
async def deploy_custom_agent(agent_id: str):
    """Deploy a custom agent"""
    try:
        from backend.registry import deploy_agent, predeploy_check
        check_result = predeploy_check(agent_id)
        if not check_result["passed"]:
            return {"success": False, "error": "Pre-deploy checks failed", "checks": check_result["checks"]}
            
        success = deploy_agent(agent_id)
        if success:
            return {"success": True, "message": f"Agent {agent_id} deployed successfully"}
        return {"success": False, "error": "Agent not found or already deployed"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.delete("/agents/{agent_id}", tags=["Agents"])
async def delete_agent(agent_id: str):
    """Delete a custom agent"""
    try:
        from backend.registry import delete_custom_agent
        success = delete_custom_agent(agent_id)
        if success:
            return {"success": True, "message": "Agent deleted"}
        return {"success": False, "error": "Agent not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/auth/login", response_model=AuthResponse, tags=["Auth"])
async def login(request: AuthRequest):
    try:
        from backend.database import verify_user, get_user_role
        if verify_user(request.username, request.password):
            role = get_user_role(request.username)
            return AuthResponse(
                success=True, 
                message="Login successful", 
                username=request.username,
                role=role,
                token=f"token-{request.username}" # Mock token for demo
            )
        return AuthResponse(success=False, message="Invalid credentials")
    except Exception as e:
        return AuthResponse(success=False, message=str(e))

@app.post("/auth/signup", response_model=AuthResponse, tags=["Auth"])
async def signup(request: AuthRequest):
    try:
        from backend.database import create_user
        success = create_user(request.username, request.password)
        if success:
            return AuthResponse(success=True, message="Account created successfully", username=request.username)
        return AuthResponse(success=False, message="Username already exists")
    except Exception as e:
        return AuthResponse(success=False, message=str(e))


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Send a message and get a response from the orchestrator.
    """
    # Pre-processing security gate
    pre = preprocess_prompt(request.message)
    if not pre["approved"]:
        raise HTTPException(status_code=400, detail=f"Request blocked: {pre['reason']}")

    try:
        result = coordinate(pre["clean_text"], request.allowed_agents)

        tasks = [
            TaskInfo(
                id=t.get("id", ""),
                description=t.get("description", ""),
                agent=t.get("agent", ""),
                status=t.get("status", ""),
                result=t.get("result", "")
            )
            for t in result.tasks
        ]

        return ChatResponse(
            response=result.final_response,
            tasks=tasks,
            agents_used=result.agents_used,
            success=result.success
        )

    except Exception as e:
        return ChatResponse(
            response=f"Error processing request: {str(e)}",
            tasks=[],
            agents_used=[],
            success=False
        )


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """
    Stream chat responses from the orchestrator.
    Returns Server-Sent Events (SSE) with real-time updates.
    """
    # Pre-processing security gate (also done inside coordinate_stream, but block early in API layer)
    pre = preprocess_prompt(request.message)
    if not pre["approved"]:
        block_reason = pre["reason"]
        block_payload = json.dumps({"type": "error", "message": f"Request blocked: {block_reason}"})

        async def blocked_stream():
            yield f"data: {block_payload}\n\n"

        return StreamingResponse(
            blocked_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    def generate():
        try:
            for update in coordinate_stream(pre["clean_text"], request.allowed_agents):
                yield f"data: {json.dumps(update)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

@app.post("/lab/chat/stream", tags=["Chat"])
async def lab_chat_stream(request: ChatRequest):
    """
    Stream chat responses from the Conversational Lab Agent.
    """
    def generate():
        try:
            from agents.lab_agent import ask_lab_stream
            for chunk in ask_lab_stream(request.message, user="frontend"):
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
            
            yield f"data: {json.dumps({'type': 'final', 'text': ''})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

@app.post("/chat/agent/{agent_id}", tags=["Chat"])
async def chat_agent_stream(agent_id: str, request: ChatRequest):
    """Stream chat responses from a specific deployed lab agent."""
    pre = preprocess_prompt(request.message)
    if not pre["approved"]:
        block_payload = json.dumps({"type": "error", "message": f"Request blocked: {pre['reason']}"})
        async def blocked_stream(): yield f"data: {block_payload}\n\n"
        return StreamingResponse(blocked_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})

    def generate():
        import time as _time
        try:
            from backend.orchestrator import analyze_query_with_constraints, execute_task, synthesize_response
            from backend.registry import get_agent
            
            agent = get_agent(agent_id)
            if not agent:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Agent {agent_id} not found'})}\n\n"
                return

            allowed_agents = agent.metadata.get('allowed_agents', ['hr', 'finance', 'scheduler', 'chart', 'sql']) if agent.metadata else None
            use_case = agent.metadata.get('responses', {}).get('problem', '') if agent.metadata else ''

            yield f"data: {json.dumps({'type': 'status', 'message': f'{agent.name} analyzing...'})}\n\n"
            
            # Simple wrapper around tasks
            tasks = analyze_query_with_constraints(pre["clean_text"], allowed_agents, use_case)
            
            for i, task in enumerate(tasks):
                yield f"data: {json.dumps({'type': 'task_start', 'agent': getattr(task, 'agent', 'unknown'), 'description': getattr(task, 'description', '')})}\n\n"
                _time.sleep(2)
                result = execute_task(task, pre["clean_text"])
                task.result = result
                yield f"data: {json.dumps({'type': 'task_complete', 'status': 'completed'})}\n\n"
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Synthesizing response'})}\n\n"
            _time.sleep(1)
            response = synthesize_response(pre["clean_text"], tasks)
            
            yield f"data: {json.dumps({'type': 'final_response', 'message': response, 'agent': agent.name})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# Run with: uvicorn backend.api:app --reload --port 8000

import os
from pathlib import Path

# Data and output paths
DATA_PATH = Path(__file__).parent.parent / "data"
OUTPUT_PATH = Path(__file__).parent.parent / "outputs"
GRAPHS_PATH = OUTPUT_PATH / "graphs"

# Ensure graphs directory exists
GRAPHS_PATH.mkdir(parents=True, exist_ok=True)

# Mount static files for graph images - allows frontend to access graphs via /graphs/{filename}
app.mount("/graphs", StaticFiles(directory=str(GRAPHS_PATH)), name="graphs")

@app.get("/graphs", tags=["Graphs"])
async def list_graphs():
    """Get a list of all generated graphs"""
    try:
        files = list(GRAPHS_PATH.glob("*.png"))
        graphs = []
        for f in sorted(files, key=os.path.getmtime, reverse=True):
            graphs.append({
                "name": f.name,
                "url": f"http://localhost:8000/graphs/{f.name}",
                "size": f.stat().st_size
            })
        return {"success": True, "graphs": graphs}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/graphs/latest", tags=["Graphs"])
async def get_latest_graph():
    """Get the URL of the most recently generated graph"""
    try:
        files = list(GRAPHS_PATH.glob("*.png"))
        if not files:
            return {"success": False, "error": "No graphs found"}
            
        # Get the most recently modified file
        latest_file = max(files, key=os.path.getmtime)
        return {"success": True, "filename": latest_file.name, "url": f"http://localhost:8000/graphs/{latest_file.name}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

class ExportRequest(BaseModel):
    """Request model for export endpoint"""
    data_type: str = "all"  # "all", "finance", "hr"
    filename: str = None


class DocumentInfo(BaseModel):
    """Model for document information"""
    name: str
    path: str
    type: str
    size: int


@app.get("/documents", response_model=list[DocumentInfo], tags=["Documents"])
async def get_documents():
    """Get list of documents from data folder (images, PDFs, Excel, CSV)"""
    documents = []
    
    if DATA_PATH.exists():
        for root, dirs, files in os.walk(DATA_PATH):
            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()
                
                if ext in ['.png', '.jpg', '.jpeg', '.gif', '.pdf', '.xlsx', '.csv', '.xls']:
                    file_type = "image" if ext in ['.png', '.jpg', '.jpeg', '.gif'] else \
                               "pdf" if ext == '.pdf' else "spreadsheet"
                    
                    documents.append(DocumentInfo(
                        name=file,
                        path=str(file_path.relative_to(DATA_PATH)),
                        type=file_type,
                        size=file_path.stat().st_size
                    ))
    
    return documents


@app.post("/documents/open", tags=["Documents"])
async def open_document(file_path: str):
    """
    Open a document file with the system's default application.
    Only works for files in the data folder.
    """
    import subprocess
    import platform
    
    try:
        # Construct full path from relative path
        full_path = DATA_PATH / file_path
        
        if not full_path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}
        
        # Security check - ensure path is within DATA_PATH
        if not str(full_path.resolve()).startswith(str(DATA_PATH.resolve())):
            return {"success": False, "error": "Access denied"}
        
        # Open with system default application
        if platform.system() == "Windows":
            os.startfile(str(full_path))
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", str(full_path)])
        else:  # Linux
            subprocess.run(["xdg-open", str(full_path)])
        
        return {"success": True, "message": f"Opened {file_path}"}
    
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/export/csv", tags=["Export"])
async def export_csv(request: ExportRequest = None):
    """Export data to CSV file"""
    from backend.tools import export_csv as do_export_csv
    import pandas as pd
    
    try:
        # Load data based on type
        if request and request.data_type == "hr":
            hr_path = DATA_PATH / "hr" / "HR Anaytics.xlsx"
            if hr_path.exists():
                df = pd.read_excel(hr_path)
            else:
                return {"success": False, "error": "HR data not found"}
        else:
            finance_path = DATA_PATH / "finance" / "Apple Financial.csv"
            if finance_path.exists():
                df = pd.read_csv(finance_path)
            else:
                return {"success": False, "error": "Finance data not found"}
        
        filename = request.filename if request and request.filename else None
        result = do_export_csv(df, filename)
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/export/notes", tags=["Export"])
async def export_notes(text: str, filename: str = None):
    """Export notes to TXT file"""
    from backend.tools import export_notes as do_export_notes
    
    result = do_export_notes(text, filename)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
