---
title: Orion Backend
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.5.0"
app_file: app.py
pinned: false
---

# Orion Multi-Agent Orchestrator

A **Multi-Agent Orchestrator System** that intelligently coordinates specialized AI agents to answer complex enterprise queries securely and accurately.

## 🚀 Architecture Overview

Orion is built as a modern, decoupled full-stack application:
- **Frontend (Next.js React SPA)**: A custom, dark-themed glassmorphic user dashboard featuring fluid Framer Motion animations, stateful user sessions via JWT/Roles, and real-time streaming using Server-Sent Events (SSE). Runs on **Port 3000**.
- **Backend (FastAPI)**: A high-performance Python server handling agent orchestration, authentication, and vector database interactions. Runs on **Port 8000**.
- **Intelligence (Mistral API)**: The Orchestrator leverages **Mistral Large** (`mistral-large-latest`) for complex reasoning, while sub-agents utilize **Mistral Nemo** (`open-mistral-nemo`) for strict tool-calling and data extraction.
- **Local Databases**: 
  - **Qdrant DB** (Vector Database for RAG context).
  - **SQLite** (User authentication and Role-Based Access Control).

## 🤖 Available Agents
- **Orchestrator** - Analyzes queries, breaks them into tasks, and coordinates sub-agents.
- **HR Agent** - Analyzes employee data, headcounts, and salaries.
- **Finance Agent** - Processes financial budgets, revenues, and transaction metadata.
- **Scheduler Agent** - Manages calendar events, meetings, and schedules.
- **Custom Agent Lab** - Allows administrators to build, test, and deploy custom agents on the fly with real-time "LLM-as-a-Judge" security and guardrail checks.

---

## 🛠️ Prerequisites

- **Python 3.10+**
- **Node.js 18+** & npm
- **Mistral API Key** (for LLM reasoning)

---

## 💻 Quick Start Guide

### Step 1: Clone & Configure

1. Clone the repository.
2. Create a `.env` file in the root directory:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

### Step 2: Set up the Backend (FastAPI)

Open Terminal 1:
```bash
# Create and activate virtual environment
python3 -m venv myenv
source myenv/bin/activate  # On Windows use: myenv\Scripts\activate

# Install requirements
pip install -r setup/requirements.txt

# Start the FastAPI Server
uvicorn backend.api:app --reload --port 8000
```

### Step 3: Set up the Frontend (Next.js)

Open Terminal 2:
```bash
# Navigate to frontend directory
cd frontend_next

# Install Node dependencies
npm install

# Start the Development Server
npm run dev
```

### Step 4: Access the Dashboard

Open your browser and navigate to: **http://localhost:3000**

*(Note: The system features Role-Based Access Control (RBAC). Upon signing up, limited users will have standard chat access, while Admin users will be able to access the Orion Agent Lab).*

---

## 🔒 Security & Data Privacy

Orion is engineered with enterprise security at its core:
1. **Domain Isolation:** Agents only access their specific functional data (e.g., HR agent cannot read Finance budgets).
2. **Pre-processing Guardrails:** Regex-based validation gates block Prompt Injections (e.g., "ignore all instructions") and SQL injections before hitting the LLM.
3. **Post-processing PII Redaction:** Automatic DLP masking of sensitive tokens and emails before delivering the response to the frontend.
4. **Local Execution:** Vector data (Qdrant) and user data (SQLite) remain 100% local. Only the sanitized prompt text is sent to the Mistral API.

---

## 📊 Testing Multi-Agent Queries

Try combining intents in the chat interface to see the Orchestrator coordinate multiple agents:
- *"Compare the Engineering headcount costs with their allocated Q1 budget."* (HR + Finance)
- *"Who has meetings scheduled for today and what are their departments?"* (HR + Scheduler)
- *"When is the next budget review meeting and what were last month's expenses?"* (Finance + Scheduler)
- *"Give me a complete overview: employee count, budget status, and upcoming meetings."* (HR + Finance + Scheduler)

## 📄 License
MIT License
