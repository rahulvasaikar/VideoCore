import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Layers, Wand2, Film, Smile, Search, FolderOpen, Play, Video, HardDrive,
  Sun, Moon, Cpu, Zap, ArrowUp, ArrowDown, Trash2, CheckCircle2, AlertCircle,
  RefreshCw, Eye, Sparkles, DownloadCloud, Image, FileVideo
} from 'lucide-react';
import './index.css';

const API_BASE = 'http://127.0.0.1:8501/api';

const TABS = [
  { id: 'joiner', label: 'Lossless Joiner', icon: Layers, desc: 'Merge files instantly without re-encoding.' },
  { id: 'enhancer', label: 'AI Enhancer', icon: Wand2, desc: 'Upscale and restore video & photo quality with Real-ESRGAN and GFPGAN.' },
  { id: 'animator', label: 'AI Animator', icon: Film, desc: 'Bidirectional stylization: Real video to Anime, or Animation to Photorealism.' },
  { id: 'facestudio', label: 'Face Studio', icon: Smile, desc: 'Scan and cluster faces in video, then preview and replace any face.' },
  { id: 'inspector', label: 'Inspector', icon: Search, desc: 'Deep dive into media codec details and stream specs.' }
];

export default function App() {
  const [activeTab, setActiveTab] = useState('joiner');
  const [theme, setTheme] = useState('dark');
  const [sysInfo, setSysInfo] = useState<any>(null);
  const [modelsStatus, setModelsStatus] = useState<any>(null);
  const [downloadingModels, setDownloadingModels] = useState(false);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const fetchSystemInfo = () => {
    fetch(`${API_BASE}/system`)
      .then(res => res.json())
      .then(data => setSysInfo(data))
      .catch(err => console.error("Could not fetch system info", err));
  };

  const fetchModelsStatus = () => {
    fetch(`${API_BASE}/models/status`)
      .then(res => res.json())
      .then(data => setModelsStatus(data))
      .catch(err => console.error("Could not fetch models status", err));
  };

  useEffect(() => {
    fetchSystemInfo();
    fetchModelsStatus();
  }, []);

  const downloadMissingModels = async () => {
    setDownloadingModels(true);
    try {
      const res = await fetch(`${API_BASE}/models/download`, { method: 'POST' });
      const { job_id } = await res.json();
      if (job_id) {
        pollJob(job_id, () => {
          setDownloadingModels(false);
          fetchModelsStatus();
        }, () => {
          setDownloadingModels(false);
        });
      }
    } catch {
      setDownloadingModels(false);
    }
  };

  const toggleTheme = () => setTheme(t => t === 'dark' ? 'light' : 'dark');

  return (
    <>
      <div className="bg-orb orb-1" />
      <div className="bg-orb orb-2" />
      
      <div className="app-layout">
        <aside className="sidebar">
          <div className="logo-container">
            <Video size={28} color="#a855f7" />
            <h1 className="logo-text">VideoCore</h1>
          </div>
          
          <nav className="nav-menu">
            {TABS.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  className={`nav-item ${isActive ? 'active' : ''}`}
                  onClick={() => setActiveTab(tab.id)}
                >
                  {isActive && (
                    <motion.div
                      layoutId="nav-indicator"
                      className="nav-indicator"
                      initial={false}
                      transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                    />
                  )}
                  <Icon size={20} />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>
          
          <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {modelsStatus && (
              <div style={{ background: 'var(--card-bg)', padding: '0.85rem', borderRadius: '12px', border: '1px solid var(--border-color)', fontSize: '0.8rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                  <span style={{ fontWeight: 600 }}>AI Model Status</span>
                  {modelsStatus.all_present ? (
                    <span className="badge" style={{ background: 'rgba(34, 197, 94, 0.2)', color: '#4ade80' }}>All Ready</span>
                  ) : (
                    <button
                      className="btn"
                      style={{ padding: '0.2rem 0.5rem', fontSize: '0.75rem' }}
                      onClick={downloadMissingModels}
                      disabled={downloadingModels}
                    >
                      <DownloadCloud size={14} /> {downloadingModels ? 'Downloading...' : 'Get Models'}
                    </button>
                  )}
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.25rem', color: 'var(--text-secondary)' }}>
                  <div>{modelsStatus.esrgan ? '✅' : '❌'} Real-ESRGAN</div>
                  <div>{modelsStatus.gfpgan ? '✅' : '❌'} GFPGAN</div>
                  <div>{modelsStatus.hayao ? '✅' : '❌'} AnimeGAN</div>
                  <div>{modelsStatus.sface ? '✅' : '❌'} SFace & YuNet</div>
                </div>
              </div>
            )}

            {sysInfo && (
              <div style={{ background: 'var(--card-bg)', padding: '0.85rem', borderRadius: '12px', border: '1px solid var(--border-color)', fontSize: '0.85rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem', fontWeight: 600 }}>
                  {sysInfo.has_nvenc ? <Zap size={16} color="#eab308" /> : <Cpu size={16} color="#94a3b8" />}
                  Hardware Acceleration
                </div>
                {sysInfo.has_nvenc ? (
                  <div style={{ color: '#4ade80', fontSize: '0.8rem' }}>NVIDIA NVENC Active</div>
                ) : (
                  <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Multi-threaded CPU Mode</div>
                )}
              </div>
            )}
            
            <button className="nav-item" onClick={toggleTheme} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', justifyContent: 'center', background: 'var(--card-bg)' }}>
              {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
              <span>{theme === 'dark' ? 'Light Mode' : 'Dark Mode'}</span>
            </button>
          </div>
        </aside>

        <main className="main-content">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.25 }}
            >
              <header className="header">
                <h2 className="title">{TABS.find(t => t.id === activeTab)?.label}</h2>
                <p className="subtitle">{TABS.find(t => t.id === activeTab)?.desc}</p>
              </header>

              <div className="content-area">
                {activeTab === 'joiner' && <JoinerTab />}
                {activeTab === 'enhancer' && <EnhancerTab />}
                {activeTab === 'animator' && <AnimatorTab />}
                {activeTab === 'facestudio' && <FaceStudioTab />}
                {activeTab === 'inspector' && <InspectorTab />}
              </div>
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </>
  );
}

// ==============================================================================
// HELPER: JOB POLLING
// ==============================================================================
function pollJob(
  jobId: string,
  onComplete: (res: any) => void,
  onError: (err: string) => void,
  onProgress?: (progress: number, stage: string, detail: string) => void
) {
  const interval = setInterval(async () => {
    try {
      const res = await fetch(`${API_BASE}/jobs/${jobId}`);
      if (!res.ok) {
        clearInterval(interval);
        onError("Could not query job status");
        return;
      }
      const data = await res.json();
      if (onProgress) {
        onProgress(data.progress || 0, data.stage || '', data.detail || '');
      }
      if (data.status === 'completed') {
        clearInterval(interval);
        onComplete(data.result);
      } else if (data.status === 'error') {
        clearInterval(interval);
        onError(data.error || 'Job failed');
      }
    } catch (e: any) {
      clearInterval(interval);
      onError(e.message || 'Job communication error');
    }
  }, 600);
}

// ==============================================================================
// TAB 1: JOINER
// ==============================================================================
function JoinerTab() {
  const [files, setFiles] = useState<string[]>([]);
  const [mode, setMode] = useState('auto');
  const [preset, setPreset] = useState('Match First Clip');
  const [outputFmt, setOutputFmt] = useState('.mp4');
  const [outPath, setOutPath] = useState('');
  
  const [browsingFolder, setBrowsingFolder] = useState(false);
  const [browsingFiles, setBrowsingFiles] = useState(false);
  const [browsingDest, setBrowsingDest] = useState(false);

  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stageText, setStageText] = useState('');
  const [detailText, setDetailText] = useState('');
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const selectFolder = async () => {
    setBrowsingFolder(true);
    try {
      const res = await fetch(`${API_BASE}/dialog/folder?title=Select Input Folder`);
      const { path } = await res.json();
      if (path) {
        const discRes = await fetch(`${API_BASE}/discover`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path })
        });
        const data = await discRes.json();
        if (data.paths && data.paths.length > 0) {
          setFiles(data.paths);
          if (!outPath) {
            const folderName = path.split('\\').pop()?.split('/').pop() || 'folder';
            setOutPath(`${path}\\${folderName}_joined${outputFmt}`);
          }
        }
      }
    } catch (e) {
      console.error(e);
    } finally {
      setBrowsingFolder(false);
    }
  };

  const selectFiles = async () => {
    setBrowsingFiles(true);
    try {
      const res = await fetch(`${API_BASE}/dialog/files?title=Select Video Files`);
      const data = await res.json();
      if (data.paths && data.paths.length > 0) {
        setFiles(prev => [...prev, ...data.paths]);
        if (!outPath) {
          const first = data.paths[0];
          const dir = first.substring(0, Math.max(first.lastIndexOf('/'), first.lastIndexOf('\\')));
          setOutPath(`${dir}\\joined_output${outputFmt}`);
        }
      }
    } catch (e) {
      console.error(e);
    } finally {
      setBrowsingFiles(false);
    }
  };

  const selectDest = async () => {
    setBrowsingDest(true);
    try {
      const res = await fetch(`${API_BASE}/dialog/save?title=Save Output As&default_name=joined_output${outputFmt}`);
      const data = await res.json();
      if (data.path) setOutPath(data.path);
    } catch (e) {
      console.error(e);
    } finally {
      setBrowsingDest(false);
    }
  };

  const moveUp = (idx: number) => {
    if (idx === 0) return;
    const newFiles = [...files];
    [newFiles[idx - 1], newFiles[idx]] = [newFiles[idx], newFiles[idx - 1]];
    setFiles(newFiles);
  };

  const moveDown = (idx: number) => {
    if (idx === files.length - 1) return;
    const newFiles = [...files];
    [newFiles[idx + 1], newFiles[idx]] = [newFiles[idx], newFiles[idx + 1]];
    setFiles(newFiles);
  };

  const removeFile = (idx: number) => {
    setFiles(files.filter((_, i) => i !== idx));
  };

  const handleExecuteJoin = async () => {
    if (files.length === 0 || !outPath) return;
    setRunning(true);
    setProgress(0);
    setStageText("Initializing video joiner...");
    setDetailText("");
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/join/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          inputs: files,
          output: outPath,
          mode,
          preset
        })
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || 'Failed to start join operation');
      }

      pollJob(
        data.job_id,
        (res) => {
          setRunning(false);
          setResult(res);
        },
        (err) => {
          setRunning(false);
          setError(err);
        },
        (pct, stage, detail) => {
          setProgress(pct);
          setStageText(stage);
          setDetailText(detail);
        }
      );
    } catch (e: any) {
      setRunning(false);
      setError(e.message || 'Operation failed');
    }
  };

  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div>
        <h3 style={{ marginTop: 0, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <FileVideo size={18} /> 1. Select Video Sequence
        </h3>
        <div className="flex-row" style={{ marginBottom: '1rem' }}>
          <button className="btn" onClick={selectFolder} disabled={running || browsingFolder}>
            <FolderOpen size={18} className={browsingFolder ? "animate-spin" : ""} /> {browsingFolder ? "Opening Explorer..." : "Add Folder"}
          </button>
          <button className="btn" onClick={selectFiles} disabled={running || browsingFiles}>
            <FileVideo size={18} className={browsingFiles ? "animate-spin" : ""} /> {browsingFiles ? "Opening Explorer..." : "Add Files"}
          </button>
          {files.length > 0 && (
            <button className="btn" onClick={() => setFiles([])} disabled={running} style={{ marginLeft: 'auto', color: '#ef4444' }}>
              <Trash2 size={16} /> Clear All
            </button>
          )}
        </div>
        
        {files.length > 0 ? (
          <div style={{ background: 'rgba(0,0,0,0.2)', borderRadius: '12px', overflow: 'hidden' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th style={{ width: '60px' }}>Order</th>
                  <th>Filename</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {files.map((f, idx) => (
                  <tr key={idx}>
                    <td style={{ color: 'var(--text-secondary)' }}>{idx + 1}</td>
                    <td style={{ wordBreak: 'break-all' }}>{f.split('\\').pop()?.split('/').pop()}</td>
                    <td style={{ textAlign: 'right', display: 'flex', gap: '0.25rem', justifyContent: 'flex-end' }}>
                      <button className="btn" style={{ padding: '0.4rem' }} disabled={running || idx === 0} onClick={() => moveUp(idx)}><ArrowUp size={16} /></button>
                      <button className="btn" style={{ padding: '0.4rem' }} disabled={running || idx === files.length - 1} onClick={() => moveDown(idx)}><ArrowDown size={16} /></button>
                      <button className="btn" style={{ padding: '0.4rem', color: '#ef4444' }} disabled={running} onClick={() => removeFile(idx)}><Trash2 size={16} /></button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{ padding: '2rem', textAlign: 'center', background: 'rgba(0,0,0,0.2)', borderRadius: '12px', color: 'var(--text-secondary)' }}>
            No videos selected. Add a folder or files to begin. You can also paste paths directly into the output box.
          </div>
        )}
      </div>

      <div>
        <h3 style={{ marginTop: 0, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Wand2 size={18} /> 2. Join Options
        </h3>
        <div className="grid-2">
          <div className="input-group">
            <label>Join Mode</label>
            <select className="input-field" value={mode} onChange={(e) => setMode(e.target.value)} disabled={running}>
              <option value="auto">Auto (Fastest Safe Choice)</option>
              <option value="fast">Fast (Strict Lossless Stream Copy)</option>
              <option value="normalize">Normalize (Re-encode Unified Canvas)</option>
            </select>
          </div>
          <div className="input-group">
            <label>Resolution Preset</label>
            <select className="input-field" value={preset} onChange={(e) => setPreset(e.target.value)} disabled={running || mode === 'fast'}>
              <option>Match First Clip</option>
              <option>1080p (1920x1080)</option>
              <option>720p (1280x720)</option>
              <option>4K (3840x2160)</option>
              <option>Vertical 1080p (1080x1920)</option>
            </select>
          </div>
          <div className="input-group">
            <label>Output Format</label>
            <select className="input-field" value={outputFmt} onChange={(e) => {
              const fmt = e.target.value;
              setOutputFmt(fmt);
              if (outPath) {
                const base = outPath.substring(0, outPath.lastIndexOf('.'));
                setOutPath(base + fmt);
              }
            }} disabled={running}>
              <option value=".mp4">.mp4</option>
              <option value=".mkv">.mkv</option>
              <option value=".mov">.mov</option>
              <option value=".avi">.avi</option>
            </select>
          </div>
          <div className="input-group">
            <label>Destination Path</label>
            <div className="flex-row">
              <input type="text" className="input-field" value={outPath} onChange={(e) => setOutPath(e.target.value)} placeholder={`G:\\Videos\\joined_output${outputFmt}`} disabled={running} />
              <button className="btn" onClick={selectDest} disabled={running || browsingDest}>
                {browsingDest ? <RefreshCw size={18} className="animate-spin" /> : <FolderOpen size={18} />}
              </button>
            </div>
          </div>
        </div>
      </div>

      {running && (
        <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1.25rem', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
            <span style={{ fontWeight: 600 }}>{stageText}</span>
            <span style={{ color: 'var(--primary)', fontWeight: 700 }}>{Math.round(progress)}%</span>
          </div>
          <div className="progress-container">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
          {detailText && <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>{detailText}</div>}
        </div>
      )}

      {error && (
        <div style={{ background: 'rgba(239, 68, 68, 0.15)', border: '1px solid #ef4444', padding: '1rem', borderRadius: '10px', color: '#fca5a5', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      {result && (
        <div className="result-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#4ade80', fontWeight: 700, fontSize: '1.1rem' }}>
            <CheckCircle2 size={22} /> Output Ready!
          </div>
          <div className="metric-grid">
            <div className="metric-box">
              <div className="metric-label">File Size</div>
              <div className="metric-val">{result.size_mb} MB</div>
            </div>
            <div className="metric-box">
              <div className="metric-label">Duration</div>
              <div className="metric-val">{result.duration.toFixed(1)}s</div>
            </div>
            <div className="metric-box">
              <div className="metric-label">Mode / Encoder</div>
              <div className="metric-val" style={{ fontSize: '1rem' }}>{result.mode} ({result.encoder})</div>
            </div>
            <div className="metric-box">
              <div className="metric-label">Time Elapsed</div>
              <div className="metric-val">{result.elapsed}s</div>
            </div>
          </div>
          <div className="flex-row">
            <button className="btn btn-primary" onClick={() => fetch(`${API_BASE}/action/play`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: result.output_path }) })}>
              <Play size={16} fill="currentColor" /> Play Video
            </button>
            <button className="btn" onClick={() => fetch(`${API_BASE}/action/explore`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: result.output_path }) })}>
              <FolderOpen size={16} /> Open in Explorer
            </button>
          </div>
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'flex-end', paddingTop: '1rem', borderTop: '1px solid var(--border-color)' }}>
        <button
          className="btn btn-primary"
          style={{ paddingLeft: '2rem', paddingRight: '2rem' }}
          disabled={running || files.length === 0 || !outPath}
          onClick={handleExecuteJoin}
        >
          <Play size={18} fill="currentColor" /> {running ? 'Processing...' : 'Execute Join'}
        </button>
      </div>
    </div>
  );
}

// ==============================================================================
// TAB 2: AI MEDIA ENHANCER
// ==============================================================================
function EnhancerTab() {
  const [mediaPath, setMediaPath] = useState('');
  const [probeData, setProbeData] = useState<any>(null);
  const [scale, setScale] = useState(2);
  const [modelName, setModelName] = useState('realesrgan-x4plus');
  const [faceRestore, setFaceRestore] = useState(true);
  const [faceFidelity, setFaceFidelity] = useState(0.70);
  const [interpolate, setInterpolate] = useState(true);
  const [codec, setCodec] = useState('h264_nvenc');
  const [outFmt, setOutFmt] = useState('.mp4');
  const [outPath, setOutPath] = useState('');

  const [browsing, setBrowsing] = useState(false);
  const [browsingDest, setBrowsingDest] = useState(false);

  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stageText, setStageText] = useState('');
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const selectMedia = async () => {
    setBrowsing(true);
    try {
      const res = await fetch(`${API_BASE}/dialog/file?title=Select Media to Enhance`);
      const { path } = await res.json();
      if (path) {
        setMediaPath(path);
        probeMedia(path);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setBrowsing(false);
    }
  };

  const probeMedia = async (p: string) => {
    try {
      const res = await fetch(`${API_BASE}/probe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: p })
      });
      const data = await res.json();
      if (data.success) {
        setProbeData(data);
        const ext = data.is_video ? '.mp4' : '.png';
        setOutFmt(ext);
        const dotIdx = p.lastIndexOf('.');
        const stem = dotIdx > 0 ? p.substring(0, dotIdx) : p;
        setOutPath(`${stem}_enhanced${ext}`);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const selectDest = async () => {
    setBrowsingDest(true);
    try {
      const res = await fetch(`${API_BASE}/dialog/save?title=Choose Enhanced Output&default_name=enhanced${outFmt}`);
      const data = await res.json();
      if (data.path) setOutPath(data.path);
    } catch (e) {
      console.error(e);
    } finally {
      setBrowsingDest(false);
    }
  };

  const handleEnhance = async () => {
    if (!mediaPath || !outPath) return;
    setRunning(true);
    setProgress(0);
    setStageText("Starting AI Enhancement Pipeline...");
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/enhance/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input: mediaPath,
          output: outPath,
          scale,
          model_name: modelName,
          face_restore: faceRestore,
          face_fidelity: faceFidelity,
          interpolate: probeData?.is_video ? interpolate : false,
          target_fps: (probeData?.is_video && interpolate) ? 60.0 : null,
          video_codec: codec,
        })
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || 'Failed to start enhancement');
      }

      pollJob(
        data.job_id,
        (res) => {
          setRunning(false);
          setResult(res);
        },
        (err) => {
          setRunning(false);
          setError(err);
        },
        (pct, stage) => {
          setProgress(pct);
          setStageText(stage);
        }
      );
    } catch (e: any) {
      setRunning(false);
      setError(e.message || 'Operation failed');
    }
  };

  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div>
        <h3 style={{ marginTop: 0, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Image size={18} /> 1. Select Media (Image or Video)
        </h3>
        <div className="flex-row">
          <input
            type="text"
            className="input-field"
            value={mediaPath}
            placeholder="Select or paste media file path..."
            onChange={(e) => {
              setMediaPath(e.target.value);
              if (e.target.value) probeMedia(e.target.value);
            }}
            disabled={running}
          />
          <button className="btn btn-primary" onClick={selectMedia} disabled={running || browsing}>
            <FolderOpen size={18} className={browsing ? "animate-spin" : ""} /> {browsing ? "Opening..." : "Browse"}
          </button>
        </div>

        {probeData && (
          <div style={{ marginTop: '1rem', background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '12px', display: 'flex', gap: '1.5rem', alignItems: 'center' }}>
            <img
              src={`${API_BASE}/media/thumbnail?path=${encodeURIComponent(mediaPath)}`}
              alt="Media thumbnail"
              style={{ width: '120px', height: '80px', objectFit: 'cover', borderRadius: '8px', background: '#000' }}
            />
            <div style={{ fontSize: '0.9rem', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem', flex: 1 }}>
              <div><strong style={{ color: 'var(--text-secondary)' }}>Type:</strong> {probeData.is_video ? 'Video' : 'Image'}</div>
              <div><strong style={{ color: 'var(--text-secondary)' }}>Resolution:</strong> {probeData.primary_video ? probeData.primary_video.resolution : 'N/A'}</div>
              <div><strong style={{ color: 'var(--text-secondary)' }}>Framerate:</strong> {probeData.primary_video ? `${probeData.primary_video.fps} FPS` : 'N/A'}</div>
              {probeData.is_video && (
                <div><strong style={{ color: 'var(--text-secondary)' }}>Duration:</strong> {probeData.duration.toFixed(1)}s</div>
              )}
              {probeData.primary_video && (
                <div><strong style={{ color: 'var(--text-secondary)' }}>Codec:</strong> {probeData.primary_video.codec}</div>
              )}
            </div>
          </div>
        )}
      </div>

      <div>
        <h3 style={{ marginTop: 0, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Sparkles size={18} /> 2. Enhancement Options
        </h3>
        <div className="grid-2">
          <div className="input-group">
            <label>Upscaling Multiplier</label>
            <select className="input-field" value={scale} onChange={(e) => setScale(Number(e.target.value))} disabled={running}>
              <option value={1}>1x (No Upscale - Face Restoration Only)</option>
              <option value={2}>2x Super-Resolution</option>
              <option value={4}>4x Ultra-Resolution</option>
            </select>
          </div>
          <div className="input-group">
            <label>Upscaling Model</label>
            <select className="input-field" value={modelName} onChange={(e) => setModelName(e.target.value)} disabled={running}>
              <option value="realesrgan-x4plus">Real-ESRGAN x4 Plus (Photos & Real World)</option>
              <option value="realesrgan-x4plus-anime">Real-ESRGAN Anime (Cartoons & Art)</option>
            </select>
          </div>
        </div>

        <div className="grid-2" style={{ marginTop: '0.5rem' }}>
          <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1.25rem', borderRadius: '12px' }}>
            <label className="checkbox-row">
              <input type="checkbox" checked={faceRestore} onChange={(e) => setFaceRestore(e.target.checked)} disabled={running} />
              <span>Enable AI Face Restoration (GFPGAN v1.4)</span>
            </label>
            <div style={{ marginTop: '1rem', opacity: faceRestore ? 1 : 0.4 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.25rem' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Identity Fidelity</span>
                <span style={{ fontWeight: 600 }}>{faceFidelity.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={faceFidelity}
                onChange={(e) => setFaceFidelity(parseFloat(e.target.value))}
                disabled={running || !faceRestore}
              />
            </div>
          </div>

          <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1.25rem', borderRadius: '12px' }}>
            {probeData?.is_video ? (
              <>
                <label className="checkbox-row">
                  <input type="checkbox" checked={interpolate} onChange={(e) => setInterpolate(e.target.checked)} disabled={running} />
                  <span>Smooth Motion to 60 FPS (RIFE Frame Interpolation)</span>
                </label>
                <div className="input-group" style={{ marginTop: '1rem', marginBottom: 0 }}>
                  <label>Encoder</label>
                  <select className="input-field" value={codec} onChange={(e) => setCodec(e.target.value)} disabled={running}>
                    <option value="h264_nvenc">H.264 (NVIDIA NVENC Fast)</option>
                    <option value="hevc_nvenc">HEVC/H.265 (NVIDIA NVENC High Efficiency)</option>
                    <option value="libx264">H.264 (CPU libx264 Fallback)</option>
                  </select>
                </div>
              </>
            ) : (
              <div style={{ color: 'var(--text-secondary)', padding: '1rem', textAlign: 'center' }}>
                Motion smoothing and video encoding apply when a video file is loaded.
              </div>
            )}
          </div>
        </div>
      </div>

      <div>
        <h3 style={{ marginTop: 0, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <FolderOpen size={18} /> 3. Destination Output
        </h3>
        <div className="flex-row">
          <input
            type="text"
            className="input-field"
            value={outPath}
            onChange={(e) => setOutPath(e.target.value)}
            placeholder="Output destination..."
            disabled={running}
          />
          <button className="btn" onClick={selectDest} disabled={running || browsingDest}>
            {browsingDest ? <RefreshCw size={18} className="animate-spin" /> : <FolderOpen size={18} />}
          </button>
        </div>
      </div>

      {running && (
        <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1.25rem', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
            <span style={{ fontWeight: 600 }}>{stageText}</span>
            <span style={{ color: 'var(--primary)', fontWeight: 700 }}>{Math.round(progress)}%</span>
          </div>
          <div className="progress-container">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}

      {error && (
        <div style={{ background: 'rgba(239, 68, 68, 0.15)', border: '1px solid #ef4444', padding: '1rem', borderRadius: '10px', color: '#fca5a5', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      {result && (
        <div className="result-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#4ade80', fontWeight: 700, fontSize: '1.1rem' }}>
            <CheckCircle2 size={22} /> AI Enhancement Complete!
          </div>
          <div className="metric-grid">
            <div className="metric-box">
              <div className="metric-label">Resolution</div>
              <div className="metric-val">{result.enh_res} <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>from {result.orig_res}</span></div>
            </div>
            {result.is_video ? (
              <div className="metric-box">
                <div className="metric-label">Framerate</div>
                <div className="metric-val">{result.enh_fps?.toFixed(1) || 'N/A'} FPS</div>
              </div>
            ) : (
              <div className="metric-box">
                <div className="metric-label">Faces Restored</div>
                <div className="metric-val">{result.faces_restored ? 'Yes' : 'No'}</div>
              </div>
            )}
            <div className="metric-box">
              <div className="metric-label">Processing Time</div>
              <div className="metric-val">{result.elapsed}s</div>
            </div>
          </div>

          {!result.is_video ? (
            <div className="grid-2" style={{ marginTop: '1rem' }}>
              <div>
                <div style={{ fontWeight: 600, marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>Original Input</div>
                <img src={`${API_BASE}/media/file?path=${encodeURIComponent(result.input)}`} alt="Original" className="preview-img" />
              </div>
              <div>
                <div style={{ fontWeight: 600, marginBottom: '0.5rem', color: '#4ade80' }}>Enhanced Output</div>
                <img src={`${API_BASE}/media/file?path=${encodeURIComponent(result.output)}`} alt="Enhanced" className="preview-img" />
              </div>
            </div>
          ) : (
            <div className="flex-row" style={{ marginTop: '1rem' }}>
              <button className="btn btn-primary" onClick={() => fetch(`${API_BASE}/action/play`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: result.output }) })}>
                <Play size={16} fill="currentColor" /> Play Enhanced Video
              </button>
              <button className="btn" onClick={() => fetch(`${API_BASE}/action/explore`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: result.output }) })}>
                <FolderOpen size={16} /> Open in Explorer
              </button>
            </div>
          )}
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'flex-end', paddingTop: '1rem', borderTop: '1px solid var(--border-color)' }}>
        <button
          className="btn btn-primary"
          style={{ paddingLeft: '2rem', paddingRight: '2rem' }}
          disabled={running || !mediaPath || !outPath}
          onClick={handleEnhance}
        >
          <Sparkles size={18} /> {running ? 'Enhancing...' : 'Run AI Enhancement'}
        </button>
      </div>
    </div>
  );
}

// ==============================================================================
// TAB 3: AI ANIMATOR (REAL <-> ANIMATION)
// ==============================================================================
function AnimatorTab() {
  const [direction, setDirection] = useState<'real_to_anim' | 'anim_to_real'>('real_to_anim');
  const [mediaPath, setMediaPath] = useState('');
  const [probeData, setProbeData] = useState<any>(null);

  // Settings
  const [style, setStyle] = useState('face_paint');
  const [strength, setStrength] = useState(1.0);
  const [lineWeight, setLineWeight] = useState(1.0);
  const [colorSaturation, setColorSaturation] = useState(1.15);
  const [edgeSoften, setEdgeSoften] = useState(0.80);
  const [filmGrain, setFilmGrain] = useState(0.15);
  const [faceRealism, setFaceRealism] = useState(true);
  const [faceFidelity, setFaceFidelity] = useState(0.85);
  const [codec, setCodec] = useState('h264_nvenc');

  const [browsing, setBrowsing] = useState(false);

  // Preview
  const [previewTime, setPreviewTime] = useState(0.0);
  const [previewOrig, setPreviewOrig] = useState<string | null>(null);
  const [previewProc, setPreviewProc] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  // Output & Run
  const [outPath, setOutPath] = useState('');
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stageText, setStageText] = useState('');
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const selectMedia = async () => {
    setBrowsing(true);
    try {
      const res = await fetch(`${API_BASE}/dialog/file?title=Select Media to Transform`);
      const { path } = await res.json();
      if (path) {
        setMediaPath(path);
        probeMedia(path);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setBrowsing(false);
    }
  };

  const probeMedia = async (p: string) => {
    try {
      const res = await fetch(`${API_BASE}/probe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: p })
      });
      const data = await res.json();
      if (data.success) {
        setProbeData(data);
        const ext = data.is_video ? '.mp4' : '.png';
        const tag = direction === 'real_to_anim' ? 'anime' : 'photoreal';
        const dotIdx = p.lastIndexOf('.');
        const stem = dotIdx > 0 ? p.substring(0, dotIdx) : p;
        setOutPath(`${stem}_${tag}${ext}`);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handlePreview = async () => {
    if (!mediaPath) return;
    setPreviewLoading(true);
    try {
      const res = await fetch(`${API_BASE}/animator/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input: mediaPath,
          direction,
          style,
          strength,
          line_weight: lineWeight,
          color_saturation: colorSaturation,
          edge_soften: edgeSoften,
          film_grain: filmGrain,
          face_realism: faceRealism,
          face_fidelity: faceFidelity,
          timestamp: previewTime,
        })
      });
      const data = await res.json();
      if (data.success) {
        setPreviewOrig(data.orig);
        setPreviewProc(data.proc);
      }
    } catch (err) {
      console.error("Preview failed", err);
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleTransform = async () => {
    if (!mediaPath || !outPath) return;
    setRunning(true);
    setProgress(0);
    setStageText("Initializing Stylization Engine...");
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/animator/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input: mediaPath,
          output: outPath,
          direction,
          style,
          strength,
          line_weight: lineWeight,
          color_saturation: colorSaturation,
          edge_soften: edgeSoften,
          film_grain: filmGrain,
          face_realism: faceRealism,
          face_fidelity: faceFidelity,
          video_codec: codec,
        })
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || 'Failed to start transformation');
      }

      pollJob(
        data.job_id,
        (res) => {
          setRunning(false);
          setResult(res);
        },
        (err) => {
          setRunning(false);
          setError(err);
        },
        (pct, stage) => {
          setProgress(pct);
          setStageText(stage);
        }
      );
    } catch (e: any) {
      setRunning(false);
      setError(e.message || 'Operation failed');
    }
  };

  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div style={{ display: 'flex', gap: '1rem', background: 'rgba(0,0,0,0.25)', padding: '0.5rem', borderRadius: '12px' }}>
        <button
          className={`btn ${direction === 'real_to_anim' ? 'btn-primary' : ''}`}
          style={{ flex: 1 }}
          onClick={() => {
            setDirection('real_to_anim');
            if (mediaPath) probeMedia(mediaPath);
          }}
          disabled={running}
        >
          🎬 Real Video / Photo ➔ Animation (Anime & Cartoon)
        </button>
        <button
          className={`btn ${direction === 'anim_to_real' ? 'btn-primary' : ''}`}
          style={{ flex: 1 }}
          onClick={() => {
            setDirection('anim_to_real');
            if (mediaPath) probeMedia(mediaPath);
          }}
          disabled={running}
        >
          🎨 Animation / Anime ➔ Real Video (Photorealism & Film)
        </button>
      </div>

      <div>
        <h3 style={{ marginTop: 0, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <FolderOpen size={18} /> 1. Select Media
        </h3>
        <div className="flex-row">
          <input
            type="text"
            className="input-field"
            value={mediaPath}
            placeholder="Select video or image file..."
            onChange={(e) => {
              setMediaPath(e.target.value);
              if (e.target.value) probeMedia(e.target.value);
            }}
            disabled={running}
          />
          <button className="btn btn-primary" onClick={selectMedia} disabled={running || browsing}>
            <FolderOpen size={18} className={browsing ? "animate-spin" : ""} /> {browsing ? "Opening..." : "Browse"}
          </button>
        </div>
      </div>

      <div>
        <h3 style={{ marginTop: 0, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Wand2 size={18} /> 2. Transformation Settings
        </h3>
        {direction === 'real_to_anim' ? (
          <div className="grid-2">
            <div className="input-group">
              <label>Animation Style Preset</label>
              <select className="input-field" value={style} onChange={(e) => setStyle(e.target.value)} disabled={running}>
                <option value="face_paint">Anime Character & Portrait (Face Paint) [AI Neural - Best for People]</option>
                <option value="hayao">Studio Ghibli (Hayao) [AI Neural - Best for Scenery]</option>
                <option value="shinkai">Makoto Shinkai (Your Name) [AI Neural - Skies & Contrast]</option>
                <option value="paprika">Satoshi Kon (Paprika) [AI Neural - Bold Fantasy]</option>
                <option value="cartoon">Cel-Shaded Cartoon [Fast Filter]</option>
                <option value="comic">Comic / Manga Ink [Fast Filter]</option>
                <option value="watercolor">Watercolor Painterly [Fast Filter]</option>
              </select>
            </div>
            <div className="input-group">
              <label>Effect Intensity ({strength.toFixed(2)})</label>
              <input type="range" min="0" max="1" step="0.05" value={strength} onChange={(e) => setStrength(parseFloat(e.target.value))} disabled={running} />
            </div>
            <div className="input-group">
              <label>Line Art Thickness ({lineWeight.toFixed(1)})</label>
              <input type="range" min="0.5" max="2.5" step="0.1" value={lineWeight} onChange={(e) => setLineWeight(parseFloat(e.target.value))} disabled={running} />
            </div>
            <div className="input-group">
              <label>Color Saturation ({colorSaturation.toFixed(2)})</label>
              <input type="range" min="0.5" max="2.0" step="0.05" value={colorSaturation} onChange={(e) => setColorSaturation(parseFloat(e.target.value))} disabled={running} />
            </div>
          </div>
        ) : (
          <div className="grid-2">
            <div className="input-group">
              <label>Delineation Factor ({edgeSoften.toFixed(2)})</label>
              <input type="range" min="0" max="1" step="0.05" value={edgeSoften} onChange={(e) => setEdgeSoften(parseFloat(e.target.value))} disabled={running} />
            </div>
            <div className="input-group">
              <label>35mm Film Grain ({filmGrain.toFixed(2)})</label>
              <input type="range" min="0" max="0.5" step="0.02" value={filmGrain} onChange={(e) => setFilmGrain(parseFloat(e.target.value))} disabled={running} />
            </div>
            <div style={{ gridColumn: 'span 2', background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '12px' }}>
              <label className="checkbox-row">
                <input type="checkbox" checked={faceRealism} onChange={(e) => setFaceRealism(e.target.checked)} disabled={running} />
                <span>Reconstruct Photorealistic Faces via GFPGAN</span>
              </label>
              {faceRealism && (
                <div style={{ marginTop: '0.75rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Face Fidelity</span>
                    <span>{faceFidelity.toFixed(2)}</span>
                  </div>
                  <input type="range" min="0.1" max="1.0" step="0.05" value={faceFidelity} onChange={(e) => setFaceFidelity(parseFloat(e.target.value))} disabled={running} />
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h3 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Eye size={18} /> 3. Instant Frame Preview
          </h3>
          <button className="btn" onClick={handlePreview} disabled={previewLoading || !mediaPath}>
            <RefreshCw size={16} className={previewLoading ? 'animate-spin' : ''} />
            {previewLoading ? 'Rendering...' : 'Render Preview'}
          </button>
        </div>

        {probeData?.is_video && probeData.duration > 0 && (
          <div className="input-group">
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Preview Timestamp</span>
              <span>{previewTime.toFixed(1)}s / {probeData.duration.toFixed(1)}s</span>
            </div>
            <input
              type="range"
              min="0"
              max={probeData.duration}
              step="0.1"
              value={previewTime}
              onChange={(e) => setPreviewTime(parseFloat(e.target.value))}
            />
          </div>
        )}

        {previewOrig && previewProc && (
          <div className="grid-2" style={{ marginTop: '1rem' }}>
            <div>
              <div style={{ color: 'var(--text-secondary)', marginBottom: '0.25rem', fontWeight: 600 }}>Original Frame</div>
              <img src={previewOrig} alt="Original Frame" className="preview-img" />
            </div>
            <div>
              <div style={{ color: 'var(--primary)', marginBottom: '0.25rem', fontWeight: 600 }}>
                Stylized ({direction === 'real_to_anim' ? style.toUpperCase() : 'PHOTOREAL'})
              </div>
              <img src={previewProc} alt="Processed Frame" className="preview-img" />
            </div>
          </div>
        )}
      </div>

      <div>
        <h3 style={{ marginTop: 0, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <FolderOpen size={18} /> 4. Destination & Encoder
        </h3>
        <div className="grid-2">
          <div className="input-group">
            <label>Output Destination</label>
            <input type="text" className="input-field" value={outPath} onChange={(e) => setOutPath(e.target.value)} disabled={running} />
          </div>
          {probeData?.is_video && (
            <div className="input-group">
              <label>Video Encoder</label>
              <select className="input-field" value={codec} onChange={(e) => setCodec(e.target.value)} disabled={running}>
                <option value="h264_nvenc">H.264 (NVIDIA NVENC Fast)</option>
                <option value="hevc_nvenc">HEVC/H.265 (NVIDIA NVENC)</option>
                <option value="libx264">H.264 (CPU libx264)</option>
              </select>
            </div>
          )}
        </div>
      </div>

      {running && (
        <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1.25rem', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
            <span style={{ fontWeight: 600 }}>{stageText}</span>
            <span style={{ color: 'var(--primary)', fontWeight: 700 }}>{Math.round(progress)}%</span>
          </div>
          <div className="progress-container">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}

      {error && (
        <div style={{ background: 'rgba(239, 68, 68, 0.15)', border: '1px solid #ef4444', padding: '1rem', borderRadius: '10px', color: '#fca5a5', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      {result && (
        <div className="result-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#4ade80', fontWeight: 700, fontSize: '1.1rem' }}>
            <CheckCircle2 size={22} /> Transformation Complete!
          </div>
          <div className="metric-grid">
            <div className="metric-box">
              <div className="metric-label">Resolution</div>
              <div className="metric-val">{result.proc_res}</div>
            </div>
            <div className="metric-box">
              <div className="metric-label">Style</div>
              <div className="metric-val" style={{ textTransform: 'capitalize' }}>{result.style}</div>
            </div>
            <div className="metric-box">
              <div className="metric-label">Time Elapsed</div>
              <div className="metric-val">{result.elapsed}s</div>
            </div>
          </div>
          <div className="flex-row">
            {result.is_video ? (
              <button className="btn btn-primary" onClick={() => fetch(`${API_BASE}/action/play`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: result.output }) })}>
                <Play size={16} fill="currentColor" /> Play Video
              </button>
            ) : null}
            <button className="btn" onClick={() => fetch(`${API_BASE}/action/explore`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: result.output }) })}>
              <FolderOpen size={16} /> Open in Explorer
            </button>
          </div>
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'flex-end', paddingTop: '1rem', borderTop: '1px solid var(--border-color)' }}>
        <button
          className="btn btn-primary"
          style={{ paddingLeft: '2rem', paddingRight: '2rem' }}
          disabled={running || !mediaPath || !outPath}
          onClick={handleTransform}
        >
          <Film size={18} /> {running ? 'Transforming...' : 'Transform Media'}
        </button>
      </div>
    </div>
  );
}

// ==============================================================================
// TAB 4: FACE STUDIO & REPLACEMENT
// ==============================================================================
function FaceStudioTab() {
  const [mediaPath, setMediaPath] = useState('');
  const [probeData, setProbeData] = useState<any>(null);

  // Scan config & clusters
  const [sampleInterval, setSampleInterval] = useState(0.5);
  const [matchThresh, setMatchThresh] = useState(0.40);
  const [scanning, setScanning] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const [scanStage, setScanStage] = useState('');
  const [clusters, setClusters] = useState<any[]>([]);
  const [analyzeJobId, setAnalyzeJobId] = useState<string | null>(null);

  // Replacement
  const [selectedPerson, setSelectedPerson] = useState<string>('');
  const [replacementPic, setReplacementPic] = useState<string>('');
  const [enableGfp, setEnableGfp] = useState(true);
  const [gfpFidelity, setGfpFidelity] = useState(0.70);
  const [blendStrength, setBlendStrength] = useState(1.0);

  // Preview
  const [previewTime, setPreviewTime] = useState(0.0);
  const [previewOrig, setPreviewOrig] = useState<string | null>(null);
  const [previewProc, setPreviewProc] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  // Browsing states
  const [browsingMedia, setBrowsingMedia] = useState(false);
  const [browsingPic, setBrowsingPic] = useState(false);

  // Full Video Execution
  const [outPath, setOutPath] = useState('');
  const [codec, setCodec] = useState('h264_nvenc');
  const [running, setRunning] = useState(false);
  const [execProgress, setExecProgress] = useState(0);
  const [execStage, setExecStage] = useState('');
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const selectMedia = async () => {
    setBrowsingMedia(true);
    try {
      const res = await fetch(`${API_BASE}/dialog/file?title=Select Video or Photo for Face Studio`);
      const { path } = await res.json();
      if (path) {
        setMediaPath(path);
        probeMedia(path);
        setClusters([]);
        setSelectedPerson('');
      }
    } catch (e) {
      console.error(e);
    } finally {
      setBrowsingMedia(false);
    }
  };

  const probeMedia = async (p: string) => {
    try {
      const res = await fetch(`${API_BASE}/probe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: p })
      });
      const data = await res.json();
      if (data.success) {
        setProbeData(data);
        const ext = data.is_video ? '.mp4' : '.png';
        const dotIdx = p.lastIndexOf('.');
        const stem = dotIdx > 0 ? p.substring(0, dotIdx) : p;
        setOutPath(`${stem}_faceswapped${ext}`);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleScanFaces = async () => {
    if (!mediaPath) return;
    setScanning(true);
    setScanProgress(0);
    setScanStage("Starting Face Scanner...");
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/face_studio/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input: mediaPath,
          sample_interval: sampleInterval,
          similarity_threshold: matchThresh,
        })
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || 'Failed to scan faces');
      setAnalyzeJobId(data.job_id);

      pollJob(
        data.job_id,
        (res) => {
          setScanning(false);
          if (res.clusters && res.clusters.length > 0) {
            setClusters(res.clusters);
            setSelectedPerson(res.clusters[0].person_id);
          }
        },
        (err) => {
          setScanning(false);
          setError(err);
        },
        (pct, stage) => {
          setScanProgress(pct);
          setScanStage(stage);
        }
      );
    } catch (e: any) {
      setScanning(false);
      setError(e.message || 'Face analysis failed');
    }
  };

  const selectReplacementPic = async () => {
    setBrowsingPic(true);
    try {
      const res = await fetch(`${API_BASE}/dialog/file?title=Select Replacement Face Photo`);
      const { path } = await res.json();
      if (path) setReplacementPic(path);
    } catch (e) {
      console.error(e);
    } finally {
      setBrowsingPic(false);
    }
  };

  const handlePreviewSwap = async () => {
    if (!mediaPath || !selectedPerson || !replacementPic) return;
    setPreviewLoading(true);
    try {
      const res = await fetch(`${API_BASE}/face_studio/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input: mediaPath,
          analyze_job_id: analyzeJobId,
          selected_person: selectedPerson,
          replacement_image: replacementPic,
          blend_strength: blendStrength,
          enable_gfpgan: enableGfp,
          gfpgan_fidelity: gfpFidelity,
          similarity_threshold: matchThresh,
          timestamp: previewTime,
        })
      });
      const data = await res.json();
      if (data.success) {
        setPreviewOrig(data.orig);
        setPreviewProc(data.proc);
      } else {
        setError(data.error || 'Preview failed');
      }
    } catch (e: any) {
      setError(e.message || 'Preview error');
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleExecuteFaceSwap = async () => {
    if (!mediaPath || !outPath || !selectedPerson || !replacementPic) return;
    setRunning(true);
    setExecProgress(0);
    setExecStage("Initializing Video Face Replacement...");
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/face_studio/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input: mediaPath,
          output: outPath,
          replacement_map: { [selectedPerson]: replacementPic },
          blend_strength: blendStrength,
          enable_gfpgan: enableGfp,
          gfpgan_fidelity: gfpFidelity,
          similarity_threshold: matchThresh,
          video_codec: codec,
        })
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || 'Face swap execution failed');

      pollJob(
        data.job_id,
        (res) => {
          setRunning(false);
          setResult(res);
        },
        (err) => {
          setRunning(false);
          setError(err);
        },
        (pct, stage) => {
          setExecProgress(pct);
          setExecStage(stage);
        }
      );
    } catch (e: any) {
      setRunning(false);
      setError(e.message || 'Operation failed');
    }
  };

  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div>
        <h3 style={{ marginTop: 0, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Smile size={18} /> 1. Select Video or Photo
        </h3>
        <div className="flex-row">
          <input
            type="text"
            className="input-field"
            value={mediaPath}
            placeholder="Select video or photo..."
            onChange={(e) => {
              setMediaPath(e.target.value);
              if (e.target.value) probeMedia(e.target.value);
            }}
            disabled={running || scanning}
          />
          <button className="btn btn-primary" onClick={selectMedia} disabled={running || scanning || browsingMedia}>
            <FolderOpen size={18} className={browsingMedia ? "animate-spin" : ""} /> {browsingMedia ? "Opening..." : "Browse"}
          </button>
        </div>
      </div>

      <div>
        <h3 style={{ marginTop: 0, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Search size={18} /> 2. Scan & Detect Faces
        </h3>
        <div className="grid-2">
          <div className="input-group">
            <label>Scanning Interval</label>
            <select className="input-field" value={sampleInterval} onChange={(e) => setSampleInterval(parseFloat(e.target.value))} disabled={scanning || running}>
              <option value={0.25}>Ultra Dense (Every 0.25s)</option>
              <option value={0.5}>Standard (Every 0.5s)</option>
              <option value={1.0}>Fast (Every 1.0s)</option>
              <option value={2.0}>High Speed (Every 2.0s)</option>
            </select>
          </div>
          <div className="input-group">
            <label>Face Clustering Sensitivity ({matchThresh.toFixed(2)})</label>
            <input type="range" min="0.30" max="0.60" step="0.05" value={matchThresh} onChange={(e) => setMatchThresh(parseFloat(e.target.value))} disabled={scanning || running} />
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-start', marginTop: '1rem' }}>
          <button className="btn btn-primary" onClick={handleScanFaces} disabled={scanning || !mediaPath}>
            <Search size={18} /> {scanning ? 'Scanning Faces...' : 'Scan & Cluster Distinct Faces'}
          </button>
        </div>

        {scanning && (
          <div style={{ marginTop: '1rem', background: 'rgba(0,0,0,0.3)', padding: '1.25rem', borderRadius: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
              <span style={{ fontWeight: 600 }}>{scanStage}</span>
              <span style={{ color: 'var(--primary)', fontWeight: 700 }}>{Math.round(scanProgress)}%</span>
            </div>
            <div className="progress-container">
              <div className="progress-fill" style={{ width: `${scanProgress}%` }} />
            </div>
          </div>
        )}

        {clusters.length > 0 && (
          <div style={{ marginTop: '1.5rem' }}>
            <div style={{ fontWeight: 600, marginBottom: '0.75rem' }}>
              Detected Faces ({clusters.length} Unique Individuals) — Click to Select Target:
            </div>
            <div className="face-grid">
              {clusters.map((c) => (
                <div
                  key={c.person_id}
                  className={`face-card ${selectedPerson === c.person_id ? 'selected' : ''}`}
                  onClick={() => {
                    setSelectedPerson(c.person_id);
                    setPreviewTime(c.first_seen_sec || 0);
                  }}
                >
                  <img src={c.thumbnail} alt={c.label} className="face-thumb" />
                  <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>{c.label}</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Seen in {c.appearance_count} frames</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {selectedPerson && (
        <div>
          <h3 style={{ marginTop: 0, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Wand2 size={18} /> 3. Replacement Picture & Settings
          </h3>
          <div className="grid-2">
            <div>
              <div className="input-group">
                <label>Replacement Face Image</label>
                <div className="flex-row">
                  <input type="text" className="input-field" value={replacementPic} placeholder="Select replacement portrait photo..." readOnly />
                  <button className="btn btn-primary" onClick={selectReplacementPic} disabled={running || browsingPic}>
                    <FolderOpen size={18} className={browsingPic ? "animate-spin" : ""} /> {browsingPic ? "Opening..." : "Browse"}
                  </button>
                </div>
              </div>
              {replacementPic && (
                <div style={{ marginTop: '0.5rem' }}>
                  <img src={`${API_BASE}/media/file?path=${encodeURIComponent(replacementPic)}`} alt="New Face" style={{ width: '100px', height: '100px', objectFit: 'cover', borderRadius: '10px', border: '1px solid var(--border-color)' }} />
                </div>
              )}
            </div>

            <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1.25rem', borderRadius: '12px' }}>
              <label className="checkbox-row">
                <input type="checkbox" checked={enableGfp} onChange={(e) => setEnableGfp(e.target.checked)} disabled={running} />
                <span>Facial Enhancement (GFPGAN)</span>
              </label>
              <div style={{ marginTop: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Fidelity</span>
                  <span>{gfpFidelity.toFixed(2)}</span>
                </div>
                <input type="range" min="0.1" max="1.0" step="0.05" value={gfpFidelity} onChange={(e) => setGfpFidelity(parseFloat(e.target.value))} disabled={running || !enableGfp} />
              </div>
              <div style={{ marginTop: '0.75rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Replacement Strength</span>
                  <span>{blendStrength.toFixed(2)}</span>
                </div>
                <input type="range" min="0.1" max="1.0" step="0.05" value={blendStrength} onChange={(e) => setBlendStrength(parseFloat(e.target.value))} disabled={running} />
              </div>
            </div>
          </div>
        </div>
      )}

      {selectedPerson && replacementPic && (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h3 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Eye size={18} /> 4. Instant Frame Preview
            </h3>
            <button className="btn" onClick={handlePreviewSwap} disabled={previewLoading}>
              <RefreshCw size={16} className={previewLoading ? 'animate-spin' : ''} />
              {previewLoading ? 'Previewing...' : 'Preview Replaced Face'}
            </button>
          </div>

          {probeData?.is_video && probeData.duration > 0 && (
            <div className="input-group">
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Preview Timestamp</span>
                <span>{previewTime.toFixed(1)}s / {probeData.duration.toFixed(1)}s</span>
              </div>
              <input
                type="range"
                min="0"
                max={probeData.duration}
                step="0.1"
                value={previewTime}
                onChange={(e) => setPreviewTime(parseFloat(e.target.value))}
              />
            </div>
          )}

          {previewOrig && previewProc && (
            <div className="grid-2" style={{ marginTop: '1rem' }}>
              <div>
                <div style={{ color: 'var(--text-secondary)', marginBottom: '0.25rem', fontWeight: 600 }}>Original Source Frame</div>
                <img src={previewOrig} alt="Original Frame" className="preview-img" />
              </div>
              <div>
                <div style={{ color: '#4ade80', marginBottom: '0.25rem', fontWeight: 600 }}>Replaced Face Frame (AI Enhanced)</div>
                <img src={previewProc} alt="Replaced Frame" className="preview-img" />
              </div>
            </div>
          )}
        </div>
      )}

      {selectedPerson && replacementPic && (
        <div>
          <h3 style={{ marginTop: 0, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <FolderOpen size={18} /> 5. Output Destination & Encoder
          </h3>
          <div className="grid-2">
            <div className="input-group">
              <label>Output Video Destination</label>
              <input type="text" className="input-field" value={outPath} onChange={(e) => setOutPath(e.target.value)} disabled={running} />
            </div>
            {probeData?.is_video && (
              <div className="input-group">
                <label>Encoder</label>
                <select className="input-field" value={codec} onChange={(e) => setCodec(e.target.value)} disabled={running}>
                  <option value="h264_nvenc">H.264 (NVIDIA NVENC Fast)</option>
                  <option value="hevc_nvenc">HEVC/H.265 (NVIDIA NVENC)</option>
                  <option value="libx264">H.264 (CPU libx264)</option>
                </select>
              </div>
            )}
          </div>
        </div>
      )}

      {running && (
        <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1.25rem', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
            <span style={{ fontWeight: 600 }}>{execStage}</span>
            <span style={{ color: 'var(--primary)', fontWeight: 700 }}>{Math.round(execProgress)}%</span>
          </div>
          <div className="progress-container">
            <div className="progress-fill" style={{ width: `${execProgress}%` }} />
          </div>
        </div>
      )}

      {error && (
        <div style={{ background: 'rgba(239, 68, 68, 0.15)', border: '1px solid #ef4444', padding: '1rem', borderRadius: '10px', color: '#fca5a5', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      {result && (
        <div className="result-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#4ade80', fontWeight: 700, fontSize: '1.1rem' }}>
            <CheckCircle2 size={22} /> Video Face Replacement Complete!
          </div>
          <div className="metric-grid">
            <div className="metric-box">
              <div className="metric-label">Resolution</div>
              <div className="metric-val">{result.resolution}</div>
            </div>
            <div className="metric-box">
              <div className="metric-label">Faces Swapped</div>
              <div className="metric-val">{result.swapped_count}</div>
            </div>
            <div className="metric-box">
              <div className="metric-label">Time Elapsed</div>
              <div className="metric-val">{result.elapsed}s</div>
            </div>
          </div>
          <div className="flex-row">
            {result.is_video ? (
              <button className="btn btn-primary" onClick={() => fetch(`${API_BASE}/action/play`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: result.output }) })}>
                <Play size={16} fill="currentColor" /> Play Video
              </button>
            ) : null}
            <button className="btn" onClick={() => fetch(`${API_BASE}/action/explore`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: result.output }) })}>
              <FolderOpen size={16} /> Open in Explorer
            </button>
          </div>
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'flex-end', paddingTop: '1rem', borderTop: '1px solid var(--border-color)' }}>
        <button
          className="btn btn-primary"
          style={{ paddingLeft: '2rem', paddingRight: '2rem' }}
          disabled={running || !selectedPerson || !replacementPic || !outPath}
          onClick={handleExecuteFaceSwap}
        >
          <Smile size={18} /> {running ? 'Replacing Faces...' : 'Update Faces Across Entire Video'}
        </button>
      </div>
    </div>
  );
}

// ==============================================================================
// TAB 5: INSPECTOR
// ==============================================================================
function InspectorTab() {
  const [file, setFile] = useState('');
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [browsing, setBrowsing] = useState(false);

  const selectFile = async () => {
    setBrowsing(true);
    try {
      const res = await fetch(`${API_BASE}/dialog/file?title=Select Media to Inspect`);
      const { path } = await res.json();
      if (path) {
        setFile(path);
        setLoading(true);
        const probeRes = await fetch(`${API_BASE}/probe`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path })
        });
        const probeData = await probeRes.json();
        if (probeData.success) {
          setData(probeData);
        }
        setLoading(false);
      }
    } catch (err) {
      console.error(err);
      setLoading(false);
    } finally {
      setBrowsing(false);
    }
  };

  return (
    <div className="glass-card">
      <div className="input-group">
        <label>Media File</label>
        <div className="flex-row">
          <input type="text" className="input-field" value={file} readOnly placeholder="No file selected..." />
          <button className="btn btn-primary" onClick={selectFile} disabled={loading || browsing}>
            <FolderOpen size={18} className={browsing ? "animate-spin" : ""} /> {browsing ? "Opening..." : "Browse"}
          </button>
        </div>
      </div>

      {loading && <div style={{ textAlign: 'center', padding: '2rem' }}>Probing media...</div>}

      {data && !loading && (
        <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }}>
          <div className="grid-2">
            <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1.5rem', borderRadius: '12px' }}>
              <div className="flex-row" style={{ justifyContent: 'space-between', marginBottom: '1rem' }}>
                <h3 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}><HardDrive size={18} /> Container</h3>
                <span className="badge">{data.format_name}</span>
              </div>
              <table className="data-table">
                <tbody>
                  <tr>
                    <td>Duration</td>
                    <td style={{ textAlign: 'right', fontWeight: 500 }}>{data.duration.toFixed(2)}s</td>
                  </tr>
                  <tr>
                    <td>Video Streams</td>
                    <td style={{ textAlign: 'right', fontWeight: 500 }}>{data.video_streams.length}</td>
                  </tr>
                  <tr>
                    <td>Audio Streams</td>
                    <td style={{ textAlign: 'right', fontWeight: 500 }}>{data.audio_streams.length}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            {data.video_streams.map((vs: any, idx: number) => (
              <div key={idx} style={{ background: 'rgba(0,0,0,0.2)', padding: '1.5rem', borderRadius: '12px' }}>
                <div className="flex-row" style={{ justifyContent: 'space-between', marginBottom: '1rem' }}>
                  <h3 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}><Video size={18} /> Video Stream {idx + 1}</h3>
                  <span className="badge">{vs.codec}</span>
                </div>
                <table className="data-table">
                  <tbody>
                    <tr>
                      <td>Resolution</td>
                      <td style={{ textAlign: 'right', fontWeight: 500 }}>{vs.resolution}</td>
                    </tr>
                    <tr>
                      <td>Framerate</td>
                      <td style={{ textAlign: 'right', fontWeight: 500 }}>{vs.fps.toFixed(2)} FPS</td>
                    </tr>
                    <tr>
                      <td>Pixel Format</td>
                      <td style={{ textAlign: 'right', fontWeight: 500 }}>{vs.pix_fmt}</td>
                    </tr>
                    <tr>
                      <td>Bitrate</td>
                      <td style={{ textAlign: 'right', fontWeight: 500 }}>{vs.bitrate ? `${(vs.bitrate / 1000).toFixed(0)} kbps` : 'N/A'}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
}
