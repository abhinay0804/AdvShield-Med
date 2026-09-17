import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ShieldCheck, Lock, User } from 'lucide-react';
import { motion } from 'framer-motion';

const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isRegistering, setIsRegistering] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  // Password Policy Checks
  const isLengthValid = password.length >= 8;
  const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(password);
  const passwordsMatch = password === confirmPassword && password !== "";
  
  // Doctor ID Policy Check
  const isDoctorIdValid = /^[A-Z]{3,5}-\d+$/.test(username);
  
  const isLoginFormValid = username.trim() !== '' && password.trim() !== '';
  const isRegisterFormValid = isDoctorIdValid && isLengthValid && hasSpecialChar && passwordsMatch;
  const isFormValid = isRegistering ? isRegisterFormValid : isLoginFormValid;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!isFormValid) return;
    
    setError('');
    setIsLoading(true);

    const endpoint = isRegistering ? '/auth/register' : '/auth/login';
    
    try {
      let response;
      if (isRegistering) {
        response = await fetch(`http://127.0.0.1:8000${endpoint}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, password }),
        });
      } else {
        const formData = new URLSearchParams();
        formData.append('username', username);
        formData.append('password', password);
        response = await fetch(`http://127.0.0.1:8000${endpoint}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: formData,
        });
      }

      if (!response.ok) {
        if (response.status === 401 && !isRegistering) {
            throw new Error("Invalid credentials entered. Please try again. If you don't have an account, please register first.");
        } else if (response.status === 400 && isRegistering) {
            throw new Error("Doctor ID already exists. Please login instead.");
        }
        throw new Error('Authentication failed. Please check your connection to the server.');
      }

      if (isRegistering) {
        setIsRegistering(false);
        setError('Registration successful! Please log in.');
        setPassword('');
        setConfirmPassword('');
      } else {
        const data = await response.json();
        localStorage.setItem('token', data.access_token);
        navigate('/dashboard');
      }
    } catch (err: any) {
      if (err.message === "Failed to fetch") {
          setError("Network error: Server is unreachable. Please ensure the backend is running.");
      } else {
          setError(err.message);
      }
    } finally {
        setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#09090b] relative overflow-hidden">
      {/* Background Glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-teal-500/10 rounded-full blur-[120px] pointer-events-none" />

      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="z-10 w-full max-w-md p-8 glass-panel"
      >
        <div className="flex flex-col items-center mb-8">
          <div className="h-16 w-16 bg-teal-500/20 text-teal-400 rounded-2xl flex items-center justify-center mb-4 border border-teal-500/30">
            <ShieldCheck size={32} />
          </div>
          <h1 className="text-2xl font-semibold tracking-tight text-white">AdvShield-Med</h1>
          <p className="text-sm text-slate-400 mt-2">Secure AI Diagnostic Portal</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <div className="relative">
              <User className="absolute left-3 top-3 h-5 w-5 text-slate-400" />
              <input
                type="text"
                placeholder={isRegistering ? "Doctor ID (e.g. RAD-1048)" : "Doctor ID"}
                value={username}
                onChange={(e) => setUsername(e.target.value.toUpperCase())}
                className="w-full bg-[#18181b] border border-white/10 rounded-lg py-2 pl-10 pr-4 text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50 transition-all"
                required
              />
            </div>
            {isRegistering && username && (
              <div className="text-xs space-y-1 mt-2 ml-1">
                <div className={`flex items-center ${isDoctorIdValid ? "text-teal-400" : "text-slate-500"}`}>
                  {isDoctorIdValid ? "✓" : "○"} Format: Specialty Prefix + Digits (e.g. RAD-1048)
                </div>
              </div>
            )}
          </div>
          
          <div className="space-y-2">
            <div className="relative">
              <Lock className="absolute left-3 top-3 h-5 w-5 text-slate-400" />
              <input
                type="password"
                placeholder={isRegistering ? "Create a strong password" : "Password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-[#18181b] border border-white/10 rounded-lg py-2 pl-10 pr-4 text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50 transition-all"
                required
              />
            </div>
            
            {isRegistering && password && (
              <div className="text-xs space-y-1 mt-2 ml-1">
                <div className={`flex items-center ${isLengthValid ? "text-teal-400" : "text-slate-500"}`}>
                  {isLengthValid ? "✓" : "○"} At least 8 characters
                </div>
                <div className={`flex items-center ${hasSpecialChar ? "text-teal-400" : "text-slate-500"}`}>
                  {hasSpecialChar ? "✓" : "○"} At least 1 special character
                </div>
              </div>
            )}
          </div>

          {isRegistering && (
            <div className="space-y-2">
              <div className="relative">
                <Lock className="absolute left-3 top-3 h-5 w-5 text-slate-400" />
                <input
                  type="password"
                  placeholder="Confirm your password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="w-full bg-[#18181b] border border-white/10 rounded-lg py-2 pl-10 pr-4 text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50 transition-all"
                  required
                />
              </div>
              {confirmPassword && (
                <div className="text-xs mt-1 ml-1">
                  {passwordsMatch ? (
                    <span className="text-teal-400 flex items-center">✓ Passwords match</span>
                  ) : (
                    <span className="text-rose-400 flex items-center">○ Passwords do not match</span>
                  )}
                </div>
              )}
            </div>
          )}

          {error && (
            <p className={`text-sm text-center ${error.includes('successful') ? 'text-teal-400' : 'text-rose-400'}`}>
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={isLoading || !isFormValid}
            className="w-full bg-teal-600 hover:bg-teal-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2 mt-2"
          >
            {isLoading ? 'Processing...' : (isRegistering ? 'Register Account' : 'Login')}
          </button>
        </form>

        <div className="mt-6 text-center">
          <button 
            onClick={() => {
              setIsRegistering(!isRegistering);
              setError('');
              setPassword('');
              setConfirmPassword('');
            }}
            className="text-sm text-slate-400 hover:text-white transition-colors"
          >
            {isRegistering ? 'Already have an account? Log in' : 'Need clinic access? Register'}
          </button>
        </div>
      </motion.div>
    </div>
  );
};

export default Login;
