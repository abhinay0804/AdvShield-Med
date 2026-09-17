import React, { useState, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { UploadCloud, ShieldAlert, ShieldCheck, Activity, FileText, LogOut, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';

interface ScanResult {
  id: number;
  patient_name: string;
  timestamp: string;
  original_image_path: string;
  heatmap_image_path: string;
  gatekeeper_flag: boolean;
  gatekeeper_confidence: number;
  diagnosis: string;
}

const Dashboard = () => {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [patientName, setPatientName] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [history, setHistory] = useState<ScanResult[]>([]);

  const token = localStorage.getItem('token');

  const fetchHistory = useCallback(async () => {
    try {
      const res = await fetch('http://127.0.0.1:8000/api/scans/history', {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setHistory(data);
      }
    } catch (e) {
      console.error(e);
    }
  }, [token]);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFile(acceptedFiles[0]);
    setResult(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png'] },
    maxFiles: 1
  });

  const handleUpload = async () => {
    if (!file || !patientName) return;
    
    setIsProcessing(true);
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/scan?patient_name=${encodeURIComponent(patientName)}`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData
      });
      
      if (!res.ok) throw new Error("Processing failed");
      
      const data = await res.json();
      setResult(data);
      fetchHistory(); // Refresh table
    } catch (error) {
      alert(error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate('/login');
  };

  return (
    <div className="min-h-screen bg-[#09090b] text-white flex">
      {/* Sidebar */}
      <div className="w-64 border-r border-white/10 p-6 flex flex-col justify-between glass-panel rounded-none">
        <div>
          <div className="flex items-center gap-3 mb-10 text-teal-400">
            <ShieldCheck size={28} />
            <h1 className="text-xl font-bold tracking-tight text-white">AdvShield</h1>
          </div>
          <nav className="space-y-2">
            <a href="#" className="flex items-center gap-3 px-4 py-3 bg-teal-500/10 text-teal-400 rounded-lg transition-colors">
              <Activity size={20} />
              <span className="font-medium">Scanner</span>
            </a>
            <a href="#" className="flex items-center gap-3 px-4 py-3 text-slate-400 hover:text-white hover:bg-white/5 rounded-lg transition-colors">
              <FileText size={20} />
              <span className="font-medium">History</span>
            </a>
          </nav>
        </div>
        <button onClick={handleLogout} className="flex items-center gap-3 px-4 py-3 text-slate-400 hover:text-red-400 transition-colors mt-auto">
          <LogOut size={20} />
          <span className="font-medium">Logout</span>
        </button>
      </div>

      {/* Main Content */}
      <div className="flex-1 p-10 overflow-y-auto">
        <header className="mb-10">
          <h2 className="text-3xl font-semibold tracking-tight">Clinical Dashboard</h2>
          <p className="text-slate-400 mt-2">Sanitize-First Adversarial Defense Pipeline active.</p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Upload Section */}
          <div className="lg:col-span-1 space-y-6">
            <div className="glass-panel p-6">
              <h3 className="text-lg font-medium mb-4">New Analysis</h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-slate-400 mb-2">Patient ID / Name</label>
                  <input
                    type="text"
                    value={patientName}
                    onChange={(e) => setPatientName(e.target.value)}
                    placeholder="e.g. PT-8921"
                    className="w-full bg-[#18181b] border border-white/10 rounded-lg py-2 px-4 text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50"
                  />
                </div>

                <div 
                  {...getRootProps()} 
                  className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all
                    ${isDragActive ? 'border-teal-500 bg-teal-500/10' : 'border-white/20 hover:border-white/40 hover:bg-white/5'}
                    ${file ? 'border-teal-500/50 bg-teal-500/5' : ''}`}
                >
                  <input {...getInputProps()} />
                  <UploadCloud className={`mx-auto h-12 w-12 mb-4 ${file ? 'text-teal-400' : 'text-slate-400'}`} />
                  <p className="text-sm text-slate-300 font-medium">
                    {file ? file.name : "Drag & drop X-Ray here"}
                  </p>
                  <p className="text-xs text-slate-500 mt-2">JPEG or PNG up to 10MB</p>
                </div>

                <button
                  onClick={handleUpload}
                  disabled={!file || !patientName || isProcessing}
                  className="w-full bg-teal-600 hover:bg-teal-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium py-3 rounded-lg transition-colors flex items-center justify-center gap-2"
                >
                  {isProcessing ? (
                    <><Loader2 className="animate-spin" size={20} /> Processing...</>
                  ) : (
                    'Run AdvShield Analysis'
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Results Section */}
          <div className="lg:col-span-2">
            <div className="glass-panel p-6 min-h-[400px] flex flex-col">
              <h3 className="text-lg font-medium mb-6">Pipeline Results</h3>
              
              <AnimatePresence mode="wait">
                {isProcessing ? (
                  <motion.div 
                    key="processing"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    className="flex-1 flex flex-col items-center justify-center text-center space-y-4"
                  >
                    <div className="relative">
                      <div className="w-24 h-24 border-4 border-teal-500/30 rounded-full"></div>
                      <div className="w-24 h-24 border-4 border-t-teal-400 rounded-full animate-spin absolute top-0 left-0"></div>
                    </div>
                    <div>
                      <p className="text-lg font-medium text-teal-400">Sanitizing Image...</p>
                      <p className="text-sm text-slate-400 mt-1">Passing through U-Net Purifier</p>
                    </div>
                  </motion.div>
                ) : result ? (
                  <motion.div 
                    key="result"
                    initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                    className="flex-1"
                  >
                    {/* Alert Banner */}
                    <div className={`p-4 rounded-lg flex items-start gap-4 mb-8 border ${result.gatekeeper_flag ? 'bg-red-500/10 border-red-500/20 text-red-400' : 'bg-teal-500/10 border-teal-500/20 text-teal-400'}`}>
                      {result.gatekeeper_flag ? <ShieldAlert size={24} className="mt-0.5 shrink-0" /> : <ShieldCheck size={24} className="mt-0.5 shrink-0" />}
                      <div>
                        <h4 className="font-semibold text-lg">
                          {result.gatekeeper_flag ? 'Adversarial Anomaly Detected!' : 'Image Validated Clean'}
                        </h4>
                        <p className="text-sm opacity-80 mt-1">
                          {result.gatekeeper_flag 
                            ? `Gatekeeper intercepted anomalous noise (Confidence: ${(result.gatekeeper_confidence * 100).toFixed(1)}%). Image was automatically routed through Purifier to ensure safe diagnosis.`
                            : 'No malicious artifacts detected. Standard diagnostic path followed.'}
                        </p>
                      </div>
                    </div>

                    {/* Images Side by Side */}
                    <div className="grid grid-cols-2 gap-6 mb-8">
                      <div>
                        <p className="text-sm text-slate-400 mb-3 font-medium">Original X-Ray</p>
                        <div className="aspect-square rounded-xl bg-black overflow-hidden border border-white/10">
                          {/* Note: In a real app we'd fetch the image via an endpoint, placeholder for now since we haven't mounted static files */}
                          <div className="w-full h-full flex items-center justify-center text-slate-500 text-sm">Image Data</div>
                        </div>
                      </div>
                      <div>
                        <p className="text-sm text-slate-400 mb-3 font-medium">Grad-CAM Heatmap (Purified)</p>
                        <div className="aspect-square rounded-xl bg-black overflow-hidden border border-white/10">
                          <div className="w-full h-full flex items-center justify-center text-slate-500 text-sm">Heatmap Data</div>
                        </div>
                      </div>
                    </div>

                    {/* Final Diagnosis */}
                    <div className="bg-[#18181b] rounded-xl p-6 border border-white/10 flex items-center justify-between">
                      <div>
                        <p className="text-sm text-slate-400 font-medium">Final Diagnosis</p>
                        <p className={`text-3xl font-bold mt-1 ${result.diagnosis === 'Pneumonia' ? 'text-rose-400' : 'text-teal-400'}`}>
                          {result.diagnosis}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-sm text-slate-400 font-medium">Patient</p>
                        <p className="text-lg font-semibold text-white mt-1">{result.patient_name}</p>
                      </div>
                    </div>

                  </motion.div>
                ) : (
                  <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">
                    Upload an X-ray to see analysis results.
                  </div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
