import { useState, useRef, useCallback } from 'react';
import PrismBackground from './components/PrismBackground';
import { T } from './i18n';
import { Upload, Leaf, AlertTriangle } from 'lucide-react';
import './index.css';

const API = import.meta.env.DEV ? 'http://localhost:5001' : '';

interface Report {
  status: string;
  plant: string;
  disease: string;
  confidence: number;
  severity: string;
  severityClass: string;
  recommendation: string;
  predictions: { class: string; confidence: number; rank: number }[];
  isHealthy: boolean;
}

const PRED_COLORS = ['#00e676', '#2979ff', '#ffea00', '#ff9100', '#ff1744',
  '#d500f9', '#00bcd4', '#8bc34a', '#ff5722', '#607d8b'];

export default function App() {
  const [lang, setLang] = useState<'en' | 'hi'>('en');
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [loading, setLoading] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const t = T[lang];

  // Check API /info on mount
  useState(() => {
    fetch(`${API}/api/info`)
      .then(r => r.json())
      .then(d => { if (!d.modelLoaded) setDemoMode(true); })
      .catch(() => setDemoMode(true));
  });

  const handleFile = useCallback((file: File) => {
    setImageFile(file);
    setImageUrl(URL.createObjectURL(file));
    setReport(null);
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleFile(file);
  }, [handleFile]);

  const analyze = async () => {
    if (!imageFile) return;
    setLoading(true);
    setReport(null);
    try {
      const form = new FormData();
      form.append('file', imageFile);
      form.append('top_k', '5');
      const resp = await fetch(`${API}/api/predict`, { method: 'POST', body: form });
      const data: Report = await resp.json();
      setReport(data);
    } catch (err) {
      console.error('Predict failed:', err);
    } finally {
      setLoading(false);
    }
  };

  const toggleLang = () => setLang(prev => (prev === 'en' ? 'hi' : 'en'));

  const formatLabel = (cls: string) => {
    const parts = cls.split('___');
    return parts.length > 1
      ? `${parts[0].replace(/_/g, ' ')} — ${parts[1].replace(/_/g, ' ')}`
      : cls;
  };

  return (
    <div className="app-wrapper">
      {/* WebGL Background */}
      <PrismBackground
        animationType="hover"
        timeScale={0.5}
        height={3.5}
        baseWidth={5.5}
        scale={3.6}
        hueShift={0}
        colorFrequency={1}
        noise={0.5}
        glow={1}
        bloom={1}
        suspendWhenOffscreen={false}
      />

      {/* Content */}
      <div className="content-layer">
        {/* ── Nav Bar ── */}
        <nav className="nav-bar">
          <div className="nav-brand">
            <span>🌿</span> {t.page_title}
          </div>
          <div className="lang-toggle" onClick={toggleLang}>
            <span className={`lang-toggle-label ${lang === 'en' ? 'active' : ''}`}>EN</span>
            <div className="toggle-track">
              <div className={`toggle-thumb ${lang === 'hi' ? 'right' : ''}`} />
            </div>
            <span className={`lang-toggle-label ${lang === 'hi' ? 'active' : ''}`}>हिंदी</span>
          </div>
        </nav>

        {/* ── Hero ── */}
        <section className="hero-card">
          <div className="hero-badge">{t.hero_badge}</div>
          <h1 className="hero-title">
            {t.hero_title_1}<br />
            <span>{t.hero_title_2}</span>
          </h1>
          <p className="hero-sub">{t.hero_sub}</p>
        </section>

        {/* ── Stats ── */}
        <div className="stats-row">
          {[
            ['38', t.disease_classes],
            ['54K+', t.training_images],
            ['224²', t.input_resolution],
            ['~95%', t.accuracy],
          ].map(([num, label]) => (
            <div className="stat-card" key={label}>
              <span className="stat-num">{num}</span>
              <span className="stat-label">{label}</span>
            </div>
          ))}
        </div>

        {/* ── Demo Notice ── */}
        {demoMode && (
          <div className="demo-notice">
            <AlertTriangle size={18} /> {t.demo_notice}
          </div>
        )}

        {/* ── Upload & Results ── */}
        <h2 className="section-title">{t.upload_title}</h2>

        <div className="upload-grid">
          {/* Left — Upload Zone */}
          <div>
            <div
              className={`upload-zone ${imageUrl ? 'has-image' : ''}`}
              onClick={() => fileRef.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={onDrop}
            >
              <input
                ref={fileRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleFile(f);
                }}
              />
              {imageUrl ? (
                <img src={imageUrl} alt={t.uploaded_caption} />
              ) : (
                <>
                  <Upload className="upload-icon" />
                  <span className="upload-hint">{t.drag_drop}</span>
                  <span className="upload-hint" style={{ fontSize: 11 }}>{t.upload_hint}</span>
                </>
              )}
            </div>
            {imageUrl && (
              <button
                className="analyze-btn"
                style={{ marginTop: 16 }}
                onClick={analyze}
                disabled={loading}
              >
                {loading ? t.analyzing : t.analyze_btn}
              </button>
            )}
          </div>

          {/* Right — Results */}
          <div className="result-panel">
            {loading ? (
              <div className="spinner">
                <div className="spinner-ring" />
                <span>{t.analyzing}</span>
              </div>
            ) : report ? (
              <>
                {/* Report Card */}
                <div className={`report-card ${report.isHealthy ? 'healthy' : 'diseased'}`}>
                  <div className="report-title">{report.plant}</div>
                  <span className={`tag ${report.isHealthy ? 'healthy' : 'diseased'}`}>
                    {report.isHealthy ? '✅' : '❌'} {report.isHealthy ? t.status_healthy : t.status_diseased}
                  </span>

                  <div style={{ marginTop: 16 }}>
                    <div className="info-row">
                      <span className="info-label">{t.disease_label}</span>
                      <span className="info-value">{report.disease}</span>
                    </div>
                    <div className="info-row">
                      <span className="info-label">{t.confidence_label}</span>
                      <span className="info-value">{(report.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="info-row">
                      <span className="info-label">{t.severity_label}</span>
                      <span className={`info-value severity-${report.severityClass}`}>{report.severity}</span>
                    </div>
                  </div>

                  <div className="rec-box">
                    <strong>{t.recommendation_label}</strong><br />
                    {report.recommendation}
                  </div>
                </div>

                {/* Confidence Bar */}
                <div style={{ marginBottom: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#546e8a', marginBottom: 4 }}>
                    <span>{t.model_confidence}</span>
                    <span style={{ color: '#e8f4ff' }}>{(report.confidence * 100).toFixed(0)}%</span>
                  </div>
                  <div className="conf-bar-wrap">
                    <div
                      className={`conf-bar-fill ${report.isHealthy ? 'green' : 'red'}`}
                      style={{ width: `${report.confidence * 100}%` }}
                    />
                  </div>
                </div>

                {/* Top Predictions */}
                <h3 className="section-title" style={{ fontSize: 14, marginBottom: 12 }}>{t.top_predictions}</h3>
                {report.predictions.map((pred) => (
                  <div className="pred-item" key={pred.rank}>
                    <div className="pred-header">
                      <span className="pred-name">#{pred.rank} &nbsp;{formatLabel(pred.class)}</span>
                      <span className="pred-pct" style={{ color: PRED_COLORS[pred.rank - 1] || '#546e8a' }}>
                        {(pred.confidence * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="conf-bar-wrap">
                      <div
                        style={{
                          width: `${pred.confidence * 100}%`,
                          background: PRED_COLORS[pred.rank - 1] || '#546e8a',
                          height: '100%',
                          borderRadius: 10,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </>
            ) : (
              <div className="result-placeholder">
                <span>{t.results_placeholder}</span>
              </div>
            )}
          </div>
        </div>

        {/* ── About ── */}
        <div style={{
          background: 'var(--card-bg)',
          border: '1px solid var(--border)',
          borderRadius: 20,
          padding: 24,
          backdropFilter: 'blur(16px)',
          display: 'flex',
          alignItems: 'center',
          gap: 16,
        }}>
          <Leaf size={32} style={{ color: 'var(--green)', opacity: 0.6, flexShrink: 0 }} />
          <p style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.8 }}>
            {t.about_model}
          </p>
        </div>

        {/* ── Footer ── */}
        <footer className="footer">{t.footer}</footer>
      </div>
    </div>
  );
}
