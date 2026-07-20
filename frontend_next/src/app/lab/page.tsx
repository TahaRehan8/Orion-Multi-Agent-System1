"use client";

import React, { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Markdown from 'markdown-to-jsx';
import { 
  Sparkles, 
  LogOut, 
  MessageSquare, 
  FlaskConical,
  Plus,
  Trash2,
  Rocket,
  AlertCircle,
  CheckCircle2,
  Cpu,
  Send,
  Bot
} from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface CustomAgent {
  id: string;
  name: string;
  description: string;
  status: string;
  tools: string[];
  prompt_template: string;
}

export default function AgentLab() {
  const router = useRouter();
  const [username, setUsername] = useState<string | null>(null);
  const [agents, setAgents] = useState<CustomAgent[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Form State
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [promptTemplate, setPromptTemplate] = useState('');
  const [allowedAgents, setAllowedAgents] = useState<string[]>([]);
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Lab Agent / AI Co-Pilot State
  const [creationMode, setCreationMode] = useState<'manual' | 'copilot'>('manual');
  const [labMessages, setLabMessages] = useState<{id: string, role: string, content: string}[]>([
    { id: 'welcome', role: 'assistant', content: '**Welcome to Orion Lab**\n\nCreate, manage, and chat with custom agents.\n\nType `customize` or click the button below to start creating a new agent.' }
  ]);
  const [labInput, setLabInput] = useState('');
  const [isLabProcessing, setIsLabProcessing] = useState(false);
  const labMessagesEndRef = useRef<HTMLDivElement>(null);

  const AVAILABLE_TOOLS = [
    { id: 'hr', label: 'HR Agent (Employee data, salary, performance)' },
    { id: 'finance', label: 'Finance Agent (Revenue, expenses, forecasts)' },
    { id: 'scheduler', label: 'Scheduler Agent (Calendar, meetings)' },
    { id: 'chart', label: 'Chart Agent (Data visualization)' },
    { id: 'sql', label: 'SQL Agent (Database queries)' }
  ];

  useEffect(() => {
    // Auth Guard
    const token = localStorage.getItem('orion_auth_token');
    const storedUsername = localStorage.getItem('orion_username');
    const role = localStorage.getItem('orion_role');
    
    if (!token || !storedUsername) {
      router.push('/login');
      return;
    }
    
    if (role !== 'super_user') {
      router.push('/');
      return;
    }
    
    setUsername(storedUsername);
    fetchAgents();
  }, [router]);

  const fetchAgents = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/agents/all`);
      const data = await res.json();
      // Filter only custom agents
      const custom = data.filter((a: any) => a.is_custom);
      setAgents(custom);
    } catch (error) {
      console.error("Failed to fetch agents", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('orion_auth_token');
    localStorage.removeItem('orion_username');
    localStorage.removeItem('orion_role');
    router.push('/login');
  };

  const toggleTool = (toolId: string) => {
    setAllowedAgents(prev => 
      prev.includes(toolId) 
        ? prev.filter(t => t !== toolId)
        : [...prev, toolId]
    );
  };

  const handleCreateAgent = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !description || !promptTemplate) {
      setError("Please fill in all required fields.");
      return;
    }

    setIsCreating(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE_URL}/agents/custom`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          description,
          tools: [], // Legacy field, we use metadata.allowed_agents now
          prompt_template: promptTemplate,
          created_by: username,
          metadata: { allowed_agents: allowedAgents }
        })
      });

      const data = await res.json();
      if (data.success) {
        // Reset form
        setName('');
        setDescription('');
        setPromptTemplate('');
        setAllowedAgents([]);
        fetchAgents();
      } else {
        setError(data.error || "Failed to create agent");
      }
    } catch (err: any) {
      setError(err.message || "An error occurred");
    } finally {
      setIsCreating(false);
    }
  };

  useEffect(() => {
    labMessagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [labMessages]);

  const handleLabSubmit = async (e?: React.FormEvent, presetCommand?: string) => {
    if (e) e.preventDefault();
    const message = presetCommand || labInput;
    if (!message.trim() || isLabProcessing) return;

    const userMessage = { id: Date.now().toString(), role: 'user', content: message };
    setLabMessages(prev => [...prev, userMessage]);
    if (!presetCommand) setLabInput('');
    setIsLabProcessing(true);

    const assistantId = (Date.now() + 1).toString();
    setLabMessages(prev => [...prev, { id: assistantId, role: 'assistant', content: '' }]);

    try {
      const response = await fetch(`${API_BASE_URL}/lab/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage.content, allowed_agents: [] })
      });

      if (!response.body) throw new Error("No response body");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let finalContent = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.substring(6));
              
              if (data.type === 'chunk') {
                finalContent += data.text;
                setLabMessages(prev => prev.map(msg => 
                  msg.id === assistantId ? { ...msg, content: finalContent } : msg
                ));
              } else if (data.type === 'final') {
                setLabMessages(prev => prev.map(msg => 
                  msg.id === assistantId ? { ...msg, content: finalContent } : msg
                ));
                // Automatically refresh agents if successful registration
                if (finalContent.includes("registered as") || finalContent.includes("successfully registered")) {
                  fetchAgents();
                }
              } else if (data.type === 'error') {
                finalContent += `\n**Error:** ${data.message}`;
                setLabMessages(prev => prev.map(msg => 
                  msg.id === assistantId ? { ...msg, content: finalContent } : msg
                ));
              }
            } catch (e) {
              // Ignore partial JSON
            }
          }
        }
      }
    } catch (error) {
      setLabMessages(prev => prev.map(msg => 
        msg.id === assistantId ? { ...msg, content: "Connection failed." } : msg
      ));
    } finally {
      setIsLabProcessing(false);
    }
  };

  const handleDeploy = async (agentId: string) => {
    try {
      const res = await fetch(`${API_BASE_URL}/agents/${agentId}/deploy`, {
        method: 'POST'
      });
      const data = await res.json();
      if (data.success) {
        fetchAgents();
      } else {
        alert("Deployment failed: " + (data.error || JSON.stringify(data.checks)));
      }
    } catch (error) {
      console.error("Failed to deploy", error);
    }
  };

  const handleDelete = async (agentId: string) => {
    if (!confirm("Are you sure you want to delete this agent?")) return;
    
    try {
      const res = await fetch(`${API_BASE_URL}/agents/${agentId}`, {
        method: 'DELETE'
      });
      const data = await res.json();
      if (data.success) {
        fetchAgents();
      }
    } catch (error) {
      console.error("Failed to delete", error);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden p-4 md:p-6 gap-6 bg-black">
      
      {/* Left Sidebar */}
      <aside className="w-64 flex-shrink-0 flex flex-col gap-4">
        <div className="glass-panel p-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl overflow-hidden bg-white/5 flex items-center justify-center border border-white/10 shadow-[0_0_15px_rgba(0,229,255,0.3)]">
              <img src="/orion_logo.png" alt="Orion Logo" className="w-full h-full object-cover" />
            </div>
            <div>
              <h1 className="font-bold tracking-tight text-white">ORION</h1>
              <p className="text-xs text-[var(--color-text-muted)] font-medium">Operator: {username}</p>
            </div>
          </div>
          <button onClick={handleLogout} className="text-red-400 hover:text-red-300" title="Logout">
            <LogOut size={16} />
          </button>
        </div>

        <div className="flex flex-col gap-1">
          <button onClick={() => router.push('/')} className="flex items-center gap-3 p-3 rounded-xl hover:bg-white/5 text-gray-400 hover:text-gray-200 font-medium transition-colors">
            <MessageSquare size={18} />
            Dashboard
          </button>
          <button className="flex items-center gap-3 p-3 rounded-xl bg-white/10 text-white font-medium">
            <FlaskConical size={18} className="text-green-400" />
            Agent Lab
          </button>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col gap-6 overflow-hidden">
        <header className="glass-panel p-6 flex-shrink-0">
          <div className="flex items-center gap-3 mb-2">
            <FlaskConical className="text-green-400" size={24} />
            <h1 className="text-2xl font-bold text-white tracking-tight">Agent Lab</h1>
          </div>
          <p className="text-[var(--color-text-muted)] max-w-3xl">
            Create, configure, and deploy custom intelligent agents. Define their specific roles, give them access to specialized sub-agents, and craft their system prompts to tailor their behavior.
          </p>
        </header>

        <div className="flex-1 flex gap-6 overflow-hidden">
          
          {/* Create Agent Form */}
          {/* Create Agent Panel */}
          <div className="glass-panel p-6 w-1/2 flex flex-col h-full overflow-hidden">
            <div className="flex items-center justify-between mb-6 flex-shrink-0">
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <Plus size={20} className="text-[var(--color-primary)]" />
                Create New Agent
              </h2>
              
              <div className="flex bg-black/40 border border-white/10 rounded-lg p-1">
                <button 
                  onClick={() => setCreationMode('manual')}
                  className={`px-3 py-1.5 text-xs font-bold rounded-md transition-all ${creationMode === 'manual' ? 'bg-[var(--color-primary)] text-black' : 'text-gray-400 hover:text-white'}`}
                >
                  Manual Form
                </button>
                <button 
                  onClick={() => setCreationMode('copilot')}
                  className={`px-3 py-1.5 text-xs font-bold rounded-md transition-all flex items-center gap-1 ${creationMode === 'copilot' ? 'bg-[var(--color-primary)] text-black' : 'text-gray-400 hover:text-white'}`}
                >
                  <Bot size={12} /> AI Co-Pilot
                </button>
              </div>
            </div>

            {creationMode === 'manual' ? (
              <div className="overflow-y-auto custom-scrollbar pr-2 pb-4">
                {error && (
                  <div className="bg-red-500/10 border border-red-500/50 text-red-400 p-3 rounded-lg text-sm mb-6 flex items-start gap-2">
                    <AlertCircle size={16} className="mt-0.5 flex-shrink-0" />
                    <p>{error}</p>
                  </div>
                )}
                <form onSubmit={handleCreateAgent} className="space-y-5">
                  <div>
                    <label className="block text-xs font-semibold text-[var(--color-text-muted)] uppercase tracking-wider mb-2">Agent Name</label>
                    <input 
                      type="text" 
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      className="w-full bg-black/40 border border-white/10 rounded-lg p-3 text-white focus:border-[var(--color-primary)] focus:ring-1 focus:ring-[var(--color-primary)] outline-none transition-all"
                      placeholder="e.g. Financial Forecaster"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-[var(--color-text-muted)] uppercase tracking-wider mb-2">Description</label>
                    <input 
                      type="text" 
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      className="w-full bg-black/40 border border-white/10 rounded-lg p-3 text-white focus:border-[var(--color-primary)] focus:ring-1 focus:ring-[var(--color-primary)] outline-none transition-all"
                      placeholder="What does this agent do?"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-[var(--color-text-muted)] uppercase tracking-wider mb-2">Allowed Sub-Agents</label>
                    <div className="space-y-2 bg-black/20 p-4 rounded-lg border border-white/5">
                      {AVAILABLE_TOOLS.map(tool => (
                        <label key={tool.id} className="flex items-center gap-3 cursor-pointer group">
                          <div className={`w-5 h-5 rounded border flex items-center justify-center transition-colors ${
                            allowedAgents.includes(tool.id) 
                              ? 'bg-[var(--color-primary)] border-[var(--color-primary)]' 
                              : 'border-gray-600 group-hover:border-gray-400'
                          }`}>
                            {allowedAgents.includes(tool.id) && <CheckCircle2 size={14} className="text-black" />}
                          </div>
                          <span className="text-sm text-gray-300 group-hover:text-white transition-colors">{tool.label}</span>
                          <input 
                            type="checkbox" 
                            className="hidden" 
                            checked={allowedAgents.includes(tool.id)}
                            onChange={() => toggleTool(tool.id)}
                          />
                        </label>
                      ))}
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-[var(--color-text-muted)] uppercase tracking-wider mb-2">System Prompt Template</label>
                    <textarea 
                      value={promptTemplate}
                      onChange={(e) => setPromptTemplate(e.target.value)}
                      className="w-full h-40 bg-black/40 border border-white/10 rounded-lg p-3 text-white focus:border-[var(--color-primary)] focus:ring-1 focus:ring-[var(--color-primary)] outline-none transition-all resize-none font-mono text-sm"
                      placeholder="You are an expert... {query}"
                    />
                    <p className="text-xs text-gray-500 mt-2">Use {'{query}'} to represent where the user's input will go.</p>
                  </div>

                  <button 
                    type="submit" 
                    disabled={isCreating}
                    className="w-full py-3 bg-[var(--color-primary)] text-black font-bold rounded-lg hover:bg-[#00e5ff] transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2"
                  >
                    {isCreating ? 'Creating...' : <><Plus size={18} /> Create Custom Agent</>}
                  </button>
                </form>
              </div>
            ) : (
              <div className="flex flex-col flex-1 overflow-hidden min-h-0">
                <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 mb-4 space-y-4">
                  {labMessages.map(msg => (
                    <div key={msg.id} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                      <div className={`p-4 rounded-xl max-w-[90%] ${msg.role === 'user' ? 'bg-white/10 text-white' : 'bg-[var(--color-primary)]/10 border border-[var(--color-primary)]/20 text-gray-200'}`}>
                        {msg.role === 'assistant' && <div className="flex items-center gap-2 mb-2 text-[var(--color-primary)] font-bold text-xs uppercase"><Bot size={14} /> Lab Agent</div>}
                        <div className="prose prose-invert prose-sm max-w-none prose-p:leading-relaxed prose-pre:bg-black/50 prose-pre:border prose-pre:border-white/10">
                          <Markdown>{msg.content}</Markdown>
                        </div>
                      </div>
                    </div>
                  ))}
                  <div ref={labMessagesEndRef} />
                </div>
                
                <div className="flex gap-2 mb-3">
                  <button onClick={() => handleLabSubmit(undefined, 'customize')} className="text-xs bg-white/5 hover:bg-white/10 text-gray-300 py-1.5 px-3 rounded border border-white/10 transition-colors">Start Customizing</button>
                  <button onClick={() => handleLabSubmit(undefined, 'list')} className="text-xs bg-white/5 hover:bg-white/10 text-gray-300 py-1.5 px-3 rounded border border-white/10 transition-colors">List Agents</button>
                  <button onClick={() => handleLabSubmit(undefined, 'cancel')} className="text-xs bg-white/5 hover:bg-white/10 text-gray-300 py-1.5 px-3 rounded border border-white/10 transition-colors">Cancel</button>
                </div>
                
                <form onSubmit={handleLabSubmit} className="flex gap-2">
                  <input 
                    type="text" 
                    value={labInput}
                    onChange={(e) => setLabInput(e.target.value)}
                    placeholder="Talk to Lab Agent..."
                    disabled={isLabProcessing}
                    className="flex-1 bg-black/40 border border-white/10 rounded-lg px-4 py-3 text-sm text-white focus:border-[var(--color-primary)] outline-none disabled:opacity-50"
                  />
                  <button 
                    type="submit" 
                    disabled={isLabProcessing || !labInput.trim()}
                    className="bg-[var(--color-primary)] text-black p-3 rounded-lg hover:bg-[#00e5ff] transition-colors disabled:opacity-50"
                  >
                    <Send size={18} />
                  </button>
                </form>
              </div>
            )}
          </div>

          {/* Existing Agents Grid */}
          <div className="glass-panel p-6 w-1/2 overflow-y-auto custom-scrollbar flex flex-col">
            <h2 className="text-lg font-bold text-white mb-6 flex items-center gap-2">
              <Cpu size={20} className="text-green-400" />
              Your Custom Agents
            </h2>

            {isLoading ? (
              <div className="flex-1 flex items-center justify-center text-[var(--color-text-muted)]">
                Loading agents...
              </div>
            ) : agents.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center text-[var(--color-text-muted)] border-2 border-dashed border-white/10 rounded-xl p-8 text-center">
                <FlaskConical size={48} className="text-gray-600 mb-4" />
                <p className="mb-2 text-gray-400 font-medium">No custom agents found</p>
                <p className="text-sm">Create an agent on the left to see it here.</p>
              </div>
            ) : (
              <div className="flex flex-col gap-4">
                {agents.map(agent => (
                  <div key={agent.id} className="bg-black/40 border border-white/10 rounded-xl p-5 hover:border-white/20 transition-all group">
                    <div className="flex justify-between items-start mb-3">
                      <div>
                        <h3 className="text-white font-bold text-lg">{agent.name}</h3>
                        <p className="text-xs text-gray-500 font-mono mt-1">ID: {agent.id}</p>
                      </div>
                      <div className={`px-2.5 py-1 text-xs font-bold rounded-full ${
                        agent.status === 'deployed' 
                          ? 'bg-green-500/20 text-green-400 border border-green-500/30' 
                          : 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
                      }`}>
                        {agent.status.toUpperCase()}
                      </div>
                    </div>
                    
                    <p className="text-sm text-gray-400 mb-4">{agent.description}</p>
                    
                    <div className="flex items-center gap-3 pt-4 border-t border-white/5 mt-auto">
                      {agent.status !== 'deployed' && (
                        <button 
                          onClick={() => handleDeploy(agent.id)}
                          className="flex-1 py-2 bg-green-500/20 hover:bg-green-500/30 text-green-400 text-sm font-semibold rounded-lg transition-colors flex justify-center items-center gap-2"
                        >
                          <Rocket size={16} /> Deploy
                        </button>
                      )}
                      <button 
                        onClick={() => handleDelete(agent.id)}
                        className="flex-1 py-2 bg-red-500/10 hover:bg-red-500/20 text-red-400 text-sm font-semibold rounded-lg transition-colors flex justify-center items-center gap-2"
                      >
                        <Trash2 size={16} /> Delete
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

        </div>
      </main>
    </div>
  );
}
