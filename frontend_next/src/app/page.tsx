'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import Markdown from 'markdown-to-jsx';
import { 
  Terminal, 
  Database, 
  BarChart3, 
  Users, 
  Calendar, 
  Send, 
  Loader2, 
  Cpu, 
  Sparkles,
  FileText,
  Download,
  LogOut,
  FolderOpen,
  FlaskConical,
  MessageSquare,
  Volume2,
  Square,
  Copy,
  Check,
  Trash2
} from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Agent {
  id: string;
  name: string;
  description: string;
}

interface DocumentInfo {
  name: string;
  path: string;
  type: string;
  size: number;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  isStreaming?: boolean;
}

const QUICK_PROMPTS = [
  "How many employees are in the Engineering department?",
  "What meetings are scheduled for January 15th?",
  "What is the total revenue for January 2025?",
  "Forecast next 3 months revenue",
  "Show me a bar chart of expenses by department",
  "Query total salary by department"
];

export default function Dashboard() {
  const router = useRouter();
  const [username, setUsername] = useState<string | null>(null);
  const [userRole, setUserRole] = useState<string | null>(null);
  
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTask, setActiveTask] = useState<string | null>(null);
  
  const [availableAgents, setAvailableAgents] = useState<Agent[]>([]);
  const [selectedAgentId, setSelectedAgentId] = useState<string>('orchestrator');
  
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [savedGraphs, setSavedGraphs] = useState<any[]>([]);
  const [exporting, setExporting] = useState(false);
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);

  const handleCopy = (id: string, content: string) => {
    navigator.clipboard.writeText(content);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const clearChat = () => {
    if (confirm("Are you sure you want to clear the chat history?")) {
      const welcomeMsg: ChatMessage = {
        id: 'welcome',
        role: 'assistant',
        content: '# System Online\nWelcome to the Orion Multi-Agent RAG Orchestrator.',
      };
      setMessages([welcomeMsg]);
      sessionStorage.setItem('orion_chat_history', JSON.stringify([welcomeMsg]));
    }
  };

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Auth Guard
    const token = localStorage.getItem('orion_auth_token');
    const user = localStorage.getItem('orion_username');
    const role = localStorage.getItem('orion_role');
    if (!token) {
      router.push('/login');
    } else {
      setUsername(user);
      setUserRole(role);
    }
  }, [router]);

  useEffect(() => {
    // Load initial data
    fetchAgents();
    fetchDocuments();
    fetchGraphs();
    
    const savedChat = sessionStorage.getItem('orion_chat_history');
    if (savedChat) {
      try {
        setMessages(JSON.parse(savedChat));
      } catch (e) {
        setMessages([
          {
            id: 'welcome',
            role: 'assistant',
            content: '# System Online\nWelcome to the Orion Multi-Agent RAG Orchestrator.',
          }
        ]);
      }
    } else {
      setMessages([
        {
          id: 'welcome',
          role: 'assistant',
          content: '# System Online\nWelcome to the Orion Multi-Agent RAG Orchestrator.',
        }
      ]);
    }
  }, []);

  // Save chat to session storage whenever it updates
  useEffect(() => {
    if (messages.length > 0) {
      sessionStorage.setItem('orion_chat_history', JSON.stringify(messages));
    }
  }, [messages]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, activeTask]);

  useEffect(() => {
    // Cleanup audio on unmount
    return () => {
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
        currentAudioRef.current.src = "";
      }
    };
  }, []);

  const toggleSpeech = async (id: string, text: string) => {
    if (playingId === id) {
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
      }
      setPlayingId(null);
    } else {
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
      }
      
      // Clean markdown text for voice
      const cleanText = text
        .replace(/!\[.*?\]\(.*?\)/g, '')
        .replace(/```[\s\S]*?```/g, 'Code block omitted.')
        .replace(/[#*`_~-]/g, '')
        .trim();
        
      // Extract the actual substance
      let textLines = cleanText.split('\n').map(l => l.trim()).filter(l => l.length > 0);
      
      // Skip the generic LLM intro sentence if it's there
      if (textLines.length > 1 && textLines[0].length < 100 && 
         /here is|based on|the data|the chart|i have|as requested/i.test(textLines[0])) {
        textLines.shift();
      }
      
      let summaryText = textLines.join('. ');
      
      // Fix date pronunciation for TTS (so it doesn't read dashes as "minus")
      let ttsText = summaryText.replace(/\b(\d{4})-(\d{2})-(\d{2})\b/g, (match, y, m, d) => {
        const months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
        return `${months[parseInt(m, 10) - 1]} ${parseInt(d, 10)}, ${y}`;
      });
      ttsText = ttsText.replace(/\b(\d{2})-(\d{2})-(\d{4})\b/g, (match, d, m, y) => {
        const months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
        return `${months[parseInt(m, 10) - 1]} ${parseInt(d, 10)}, ${y}`;
      });
      
      // Strip out raw URLs
      ttsText = ttsText.replace(/https?:\/\/[^\s]+/g, '');
      
      // Strip out ugly backend filenames or absolute paths
      ttsText = ttsText.replace(/graph_[a-z]+_\d+_\d+\.png/g, 'the generated chart');
      ttsText = ttsText.replace(/\/(home|opt|usr|var)\/[^\s]+/g, '');
      ttsText = ttsText.replace(/Chart saved locally at:\s*/g, 'Chart saved. ');
        
      console.log("[Speech] Attempting to synthesize backend text:", ttsText);
      
      if (!ttsText) {
        console.warn("[Speech] No text remaining to read.");
        return;
      }
        
      setPlayingId(id);
      
      try {
        // By passing the URL directly to the Audio object, the browser handles 
        // the chunked HTTP stream natively and starts playing instantly!
        const audioUrl = `${API_BASE_URL}/tts?text=${encodeURIComponent(ttsText)}`;
        const audio = new Audio(audioUrl);
        
        // Increase playback speed slightly for a snappier feel
        audio.playbackRate = 1.2;
        
        audio.onended = () => setPlayingId(null);
        audio.onerror = (e) => {
          console.error("[Speech] Audio playback error", e);
          setPlayingId(null);
        };
        
        currentAudioRef.current = audio;
        audio.play();
        
      } catch (e) {
        console.error("[Speech] Network or API error:", e);
        setPlayingId(null);
      }
    }
  };

  const fetchAgents = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/agents`);
      const data = await res.json();
      setAvailableAgents(data);
    } catch (e) {
      console.error("Failed to load agents", e);
    }
  };

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/documents`);
      const data = await res.json();
      setDocuments(data);
    } catch (e) {
      console.error("Error fetching documents:", e);
    }
  };

  const fetchGraphs = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/graphs`);
      const data = await res.json();
      if (data.success) setSavedGraphs(data.graphs);
    } catch (e) {
      console.error("Error fetching graphs:", e);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('orion_auth_token');
    localStorage.removeItem('orion_username');
    localStorage.removeItem('orion_role');
    router.push('/login');
  };

  const openDocument = async (path: string) => {
    try {
      await fetch(`${API_BASE_URL}/documents/open?file_path=${encodeURIComponent(path)}`, {
        method: 'POST'
      });
    } catch (e) {
      console.error("Failed to open document", e);
    }
  };

  const handleExport = async (type: 'finance' | 'hr') => {
    setExporting(true);
    try {
      const res = await fetch(`${API_BASE_URL}/export/csv`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data_type: type })
      });
      const data = await res.json();
      if (data.success && data.url) {
        forceDownload(data.url, data.filename || `export_${type}.csv`);
      } else {
        alert(`Export failed: ${data.error || 'Unknown error'}`);
      }
    } catch (e) {
      console.error("Export error", e);
    } finally {
      setExporting(false);
    }
  };

  const forceDownload = async (url: string, filename: string) => {
    try {
      const response = await fetch(url);
      const blob = await response.blob();
      const blobUrl = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = blobUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(blobUrl);
    } catch (e) {
      console.error("Download failed, opening in new tab", e);
      window.open(url, '_blank');
    }
  };

  const handleDownloadLatestChart = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/graphs/latest`);
      const data = await res.json();
      if (data.success && data.url) {
        forceDownload(data.url, data.filename || 'chart.png');
      } else {
        alert(data.error || 'No charts available to download yet.');
      }
    } catch (e) {
      console.error("Error downloading chart:", e);
      alert('Failed to connect to backend.');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isProcessing) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsProcessing(true);
    setActiveTask('Initializing stream...');

    const assistantId = (Date.now() + 1).toString();
    setMessages(prev => [...prev, {
      id: assistantId,
      role: 'assistant',
      content: '',
      isStreaming: true,
    }]);

    try {
      // Clear input and start processing
      setInput('');
      setIsProcessing(true);
      setActiveTask('Initializing stream...');

      const endpoint = selectedAgentId === 'orchestrator' 
        ? `${API_BASE_URL}/chat/stream` 
        : `${API_BASE_URL}/chat/agent/${selectedAgentId}`;

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage.content })
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
              
              if (data.type === 'status' || data.type === 'task_start') {
                setActiveTask(data.message || data.description || 'Processing...');
              } 
              else if (data.type === 'final_response') {
                finalContent = data.message;
                setMessages(prev => prev.map(msg => 
                  msg.id === assistantId ? { ...msg, content: finalContent, isStreaming: false } : msg
                ));
                setActiveTask(null);
                fetchGraphs();
              }
              else if (data.type === 'final') {
                finalContent = data.response;
                setMessages(prev => prev.map(msg => 
                  msg.id === assistantId ? { ...msg, content: finalContent, isStreaming: false } : msg
                ));
                setActiveTask(null);
                fetchGraphs();
              }
              else if (data.type === 'error') {
                finalContent = `**Error:** ${data.message}`;
                setMessages(prev => prev.map(msg => 
                  msg.id === assistantId ? { ...msg, content: finalContent, isStreaming: false } : msg
                ));
                setActiveTask(null);
              }
            } catch (e) {
              // Ignore partial JSON
            }
          }
        }
      }
    } catch (error) {
      setMessages(prev => prev.map(msg => 
        msg.id === assistantId ? { ...msg, content: "Connection failed.", isStreaming: false } : msg
      ));
      setActiveTask(null);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden p-4 md:p-6 gap-6 bg-black">
      
      {/* Left Sidebar: Agents & Lab */}
      <aside className="w-64 flex-shrink-0 flex flex-col gap-4">
        <div className="glass-panel p-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-[var(--color-primary-dim)] flex items-center justify-center overflow-hidden neon-glow-primary border border-[var(--color-primary)]/30">
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
          <button className="flex items-center gap-3 p-3 rounded-xl bg-white/10 text-white font-medium">
            <MessageSquare size={18} className="text-[var(--color-primary)]" />
            Dashboard
          </button>
          {userRole === 'super_user' && (
            <button onClick={() => router.push('/lab')} className="flex items-center gap-3 p-3 rounded-xl hover:bg-white/5 text-gray-400 hover:text-gray-200 font-medium transition-colors">
              <FlaskConical size={18} className="text-green-400" />
              Agent Lab
            </button>
          )}
        </div>

        <div className="glass-panel p-4 flex-1 flex flex-col gap-4 overflow-y-auto">
          <div>
            <h2 className="text-xs font-bold text-[var(--color-text-muted)] uppercase tracking-wider mb-2">Agent Routing</h2>
            <select 
              value={selectedAgentId} 
              onChange={(e) => setSelectedAgentId(e.target.value)}
              className="w-full bg-black/40 border border-white/10 rounded-lg p-2 text-white text-sm outline-none focus:border-[var(--color-primary)]"
            >
              {availableAgents.map(a => (
                <option key={a.id} value={a.id}>{a.name}</option>
              ))}
            </select>
            <p className="text-xs text-gray-400 mt-2">
              {availableAgents.find(a => a.id === selectedAgentId)?.description}
            </p>
          </div>
          
          <div className="mt-4">
             <h2 className="text-xs font-bold text-[var(--color-text-muted)] uppercase tracking-wider mb-2">System Status</h2>
             <div className="flex items-center gap-2 text-sm text-green-400 p-2 bg-green-500/10 rounded-lg border border-green-500/20 mb-3">
               <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
               Nodes Online
             </div>
             
             <h2 className="text-xs font-bold text-[var(--color-text-muted)] uppercase tracking-wider mb-2 mt-4">Chart Tools</h2>
             <button 
               onClick={handleDownloadLatestChart}
               className="w-full flex items-center gap-2 text-sm font-medium bg-[var(--color-primary)]/10 text-[var(--color-primary)] border border-[var(--color-primary)]/20 py-2 px-3 rounded-lg hover:bg-[var(--color-primary)]/20 transition-colors justify-between"
             >
               <span className="flex items-center gap-2">
                 <Download size={14} /> Download Last Chart
               </span>
             </button>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col gap-4 min-w-0">
        
        {/* Messages */}
        <div className="glass-panel flex-1 p-6 overflow-y-auto flex flex-col gap-6 scroll-smooth">
          {messages.map((msg) => (
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              key={msg.id} 
              className={`flex flex-col max-w-[85%] ${msg.role === 'user' ? 'self-end items-end' : 'self-start items-start'}`}
            >
              <div className="flex items-center gap-2 mb-2 px-1">
                {msg.role === 'assistant' ? (
                  <>
                    <div className="flex items-center gap-2 flex-1">
                      <Cpu size={16} className="text-[var(--color-primary)]" />
                      <span className="text-xs font-semibold text-[var(--color-text-muted)] tracking-wider uppercase">
                        {selectedAgentId === 'orchestrator' ? 'System Output' : 'Lab Agent Output'}
                      </span>
                    </div>
                    {msg.content && !msg.isStreaming && (
                      <div className="flex items-center gap-1">
                        <button 
                          onClick={() => handleCopy(msg.id, msg.content)}
                          className="p-1.5 rounded-md transition-colors hover:bg-white/10 text-gray-400 hover:text-white"
                          title="Copy to clipboard"
                        >
                          {copiedId === msg.id ? <Check size={14} className="text-green-400" /> : <Copy size={14} />}
                        </button>
                        <button 
                          onClick={() => toggleSpeech(msg.id, msg.content)}
                          className={`p-1.5 rounded-md transition-colors ${
                            playingId === msg.id 
                              ? 'bg-[var(--color-secondary-dim)] text-[var(--color-secondary)]' 
                              : 'hover:bg-white/10 text-gray-400 hover:text-white'
                          }`}
                          title={playingId === msg.id ? "Stop reading" : "Read aloud"}
                        >
                          {playingId === msg.id ? <Square size={14} fill="currentColor" /> : <Volume2 size={14} />}
                        </button>
                      </div>
                    )}
                  </>
                ) : (
                  <span className="text-xs font-semibold text-[var(--color-secondary)] tracking-wider uppercase">User</span>
                )}
              </div>
              
              <div className={`p-4 rounded-2xl min-w-0 max-w-full overflow-x-auto ${msg.role === 'user' ? 'bg-[var(--color-secondary-dim)] border border-[var(--color-secondary)]/30 text-white rounded-tr-sm' : 'bg-white/5 border border-white/10 text-gray-200 rounded-tl-sm shadow-xl'}`}>
                {msg.role === 'assistant' && msg.isStreaming && !msg.content ? (
                  <div className="flex items-center gap-3 text-cyan-200">
                    <Loader2 size={18} className="animate-spin" />
                    <span className="text-sm italic">{activeTask}</span>
                  </div>
                ) : (
                  <div className="text-sm md:text-base space-y-4 w-full min-w-0 break-words">
                    <Markdown 
                      options={{
                        overrides: {
                          img: {
                            props: {
                              className: "max-w-full max-h-[400px] object-contain rounded-lg mx-auto my-4 border border-white/10"
                            }
                          },
                          p: {
                            props: {
                              className: "leading-relaxed break-words whitespace-pre-wrap"
                            }
                          },
                          pre: {
                            props: {
                              className: "bg-black/50 border border-white/10 p-4 rounded-lg overflow-x-auto my-2 max-w-full"
                            }
                          },
                          table: {
                            props: {
                              className: "w-full border-collapse my-4 block overflow-x-auto"
                            }
                          },
                          th: {
                            props: {
                              className: "border border-white/20 p-3 bg-white/5 whitespace-nowrap"
                            }
                          },
                          td: {
                            props: {
                              className: "border border-white/10 p-3"
                            }
                          }
                        }
                      }}
                    >
                      {msg.content}
                    </Markdown>
                  </div>
                )}
              </div>
            </motion.div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Quick Prompts */}
        <div className="px-4 no-print">
          <div className="flex justify-between items-center mb-2 ml-1">
            <h3 className="text-xs font-semibold text-[var(--color-text-muted)] uppercase tracking-wider">Quick Prompts</h3>
            <button 
              onClick={clearChat}
              className="text-xs flex items-center gap-1 text-red-400/80 hover:text-red-300 transition-colors"
              title="Clear entire chat history"
            >
              <Trash2 size={12} /> Clear Chat
            </button>
          </div>
          <div className="flex flex-wrap gap-2">
            {QUICK_PROMPTS.map((prompt, idx) => (
              <button 
                key={idx}
                onClick={() => setInput(prompt)}
                className="text-xs bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 py-2 px-3 rounded-lg transition-colors text-left truncate max-w-[250px]"
                title={prompt}
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>

        {/* Input Area */}
        <div className="glass-panel p-2 pl-4 flex items-center gap-4 group no-print">
          <input 
            type="text" 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSubmit(e)}
            placeholder={`Send message to ${availableAgents.find(a => a.id === selectedAgentId)?.name || 'System'}...`}
            disabled={isProcessing}
            className="flex-1 bg-transparent border-none outline-none text-white placeholder-[var(--color-text-muted)] h-12 font-medium"
          />
          <button 
            onClick={handleSubmit}
            disabled={isProcessing || !input.trim()}
            className="h-12 w-12 rounded-xl bg-[var(--color-primary)] flex items-center justify-center text-black disabled:opacity-50 disabled:bg-gray-600 transition-all hover:scale-105 active:scale-95"
          >
            <Send size={20} />
          </button>
        </div>

      </main>

      {/* Right Sidebar: Documents & Exports */}
      <aside className="w-64 flex-shrink-0 flex flex-col gap-4">
        <div className="glass-panel p-4 flex-1 flex flex-col gap-6 overflow-y-auto">
          
          <div className="border-b border-white/10 pb-6">
            <h2 className="text-xs font-bold text-[var(--color-text-muted)] uppercase tracking-wider mb-3 flex items-center gap-2">
              <Download size={14} /> Export Tools
            </h2>
            <div className="flex flex-col gap-2">
              <button 
                onClick={() => handleExport('finance')}
                disabled={exporting}
                className="w-full text-xs font-medium bg-yellow-500/10 text-yellow-400 border border-yellow-500/20 py-2 px-3 rounded-lg hover:bg-yellow-500/20 transition-colors text-left flex justify-between items-center"
              >
                Export Finance Data
                {exporting && <Loader2 size={12} className="animate-spin" />}
              </button>
              <button 
                onClick={() => window.print()}
                className="w-full text-xs font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20 py-2 px-3 rounded-lg hover:bg-blue-500/20 transition-colors text-left flex justify-between items-center"
              >
                Export Chat as PDF
              </button>
              <button 
                onClick={() => handleExport('hr')}
                disabled={exporting}
                className="w-full text-xs font-medium bg-purple-500/10 text-purple-400 border border-purple-500/20 py-2 px-3 rounded-lg hover:bg-purple-500/20 transition-colors text-left flex justify-between items-center"
              >
                Export HR Data
                {exporting && <Loader2 size={12} className="animate-spin" />}
              </button>
            </div>
          </div>

          <div>
            <h2 className="text-xs font-bold text-[var(--color-text-muted)] uppercase tracking-wider mb-3 flex items-center gap-2">
              <FolderOpen size={14} /> Knowledge Base
            </h2>
            <div className="flex flex-col gap-2">
              {documents.length === 0 ? (
                <p className="text-xs text-gray-500 italic">No files found.</p>
              ) : (
                documents.map(doc => (
                  <button 
                    key={doc.path} 
                    onClick={() => openDocument(doc.path)}
                    className="flex items-center gap-2 text-left p-2 rounded-lg hover:bg-white/5 transition-colors border border-transparent hover:border-white/10"
                  >
                    <FileText size={16} className="text-blue-400 flex-shrink-0" />
                    <div className="overflow-hidden">
                      <p className="text-sm text-gray-200 truncate">{doc.name}</p>
                      <p className="text-xs text-gray-500">{(doc.size / 1024).toFixed(1)} KB</p>
                    </div>
                  </button>
                ))
              )}
            </div>
          </div>

          <div className="mt-6">
            <h2 className="text-xs font-bold text-[var(--color-text-muted)] uppercase tracking-wider mb-3 flex items-center gap-2">
              <BarChart3 size={14} /> Saved Charts
            </h2>
            <div className="flex flex-col gap-2">
              {savedGraphs.length === 0 ? (
                <p className="text-xs text-gray-500 italic">No charts found.</p>
              ) : (
                savedGraphs.map(graph => (
                  <button 
                    key={graph.name} 
                    onClick={() => forceDownload(graph.url, graph.name)}
                    className="flex items-center gap-2 text-left p-2 rounded-lg hover:bg-white/5 transition-colors border border-transparent hover:border-white/10"
                    title="Click to download"
                  >
                    <BarChart3 size={16} className="text-cyan-400 flex-shrink-0" />
                    <div className="overflow-hidden">
                      <p className="text-sm text-gray-200 truncate">{graph.name}</p>
                      <p className="text-xs text-gray-500">{(graph.size / 1024).toFixed(1)} KB</p>
                    </div>
                  </button>
                ))
              )}
            </div>
          </div>

        </div>
      </aside>

    </div>
  );
}
