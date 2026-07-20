'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { Sparkles, Loader2, Lock, User, AlertCircle } from 'lucide-react';

export default function LoginPage() {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const router = useRouter();

  // Redirect to dashboard if already logged in
  useEffect(() => {
    if (localStorage.getItem('orion_auth_token')) {
      router.push('/');
    }
  }, [router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !password.trim() || (!isLogin && !confirmPassword.trim())) {
      setError('Please fill in all fields');
      return;
    }
    
    if (!isLogin && password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const endpoint = isLogin ? '/auth/login' : '/auth/signup';
      const response = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });

      const data = await response.json();

      if (data.success) {
        if (isLogin) {
          localStorage.setItem('orion_auth_token', data.token);
          localStorage.setItem('orion_username', data.username);
          localStorage.setItem('orion_role', data.role);
          router.push('/');
        } else {
          // Signup successful, switch to login
          setIsLogin(true);
          setError('');
          alert('Account created successfully! Please log in.');
        }
      } else {
        setError(data.message || 'Authentication failed');
      }
    } catch (err) {
      setError('Cannot connect to backend server');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center p-4">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-panel w-full max-w-md p-8 flex flex-col gap-6"
      >
        <div className="text-center flex flex-col items-center gap-3 mb-4">
          <div className="w-16 h-16 rounded-2xl overflow-hidden bg-white/5 flex items-center justify-center border border-white/10 shadow-[0_0_15px_rgba(0,229,255,0.3)]">
            <img src="/orion_logo.png" alt="Orion Logo" className="w-full h-full object-cover" />
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-white mt-2">
            {isLogin ? 'Sign In' : 'Sign Up'}
          </h1>
          <p className="text-sm text-[var(--color-text-muted)]">
            Orion Multi-Agent RAG System
          </p>
        </div>

        {error && (
          <div className="bg-red-500/10 border border-red-500/30 text-red-400 p-3 rounded-lg text-sm flex items-center gap-2">
            <AlertCircle size={16} />
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div className="relative">
            <User size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--color-text-muted)]" />
            <input
              type="text"
              placeholder="Operator ID"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full bg-black/40 border border-white/10 rounded-xl py-3 pl-10 pr-4 text-white focus:outline-none focus:border-[var(--color-primary)] transition-colors placeholder:text-gray-600"
            />
          </div>

          <div className="relative">
            <Lock size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--color-text-muted)]" />
            <input
              type="password"
              placeholder="Passcode"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-black/40 border border-white/10 rounded-xl py-3 pl-10 pr-4 text-white focus:outline-none focus:border-[var(--color-primary)] transition-colors placeholder:text-gray-600"
            />
          </div>

          {!isLogin && (
            <div className="relative">
              <Lock size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--color-text-muted)]" />
              <input
                type="password"
                placeholder="Confirm Passcode"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                className="w-full bg-black/40 border border-white/10 rounded-xl py-3 pl-10 pr-4 text-white focus:outline-none focus:border-[var(--color-primary)] transition-colors placeholder:text-gray-600"
              />
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full mt-4 bg-[var(--color-primary)] hover:bg-[var(--color-primary)]/80 text-black font-semibold rounded-xl py-3 flex items-center justify-center gap-2 transition-all hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:pointer-events-none neon-glow-primary"
          >
            {loading ? <Loader2 size={18} className="animate-spin" /> : (isLogin ? 'Authenticate' : 'Initialize Protocol')}
          </button>
        </form>

        <div className="text-center mt-2">
          <button 
            type="button"
            onClick={() => setIsLogin(!isLogin)}
            className="text-sm text-[var(--color-text-muted)] hover:text-white transition-colors"
          >
            {isLogin ? 'No active credentials? Request access.' : 'Already have clearance? Authenticate.'}
          </button>
        </div>
      </motion.div>
    </div>
  );
}
