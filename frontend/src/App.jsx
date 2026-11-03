import { useEffect, useRef, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import { uploadImage, getAlerts, getHistory, getModelInfo } from './services/api';
import { normalizeAnalysisResult } from './services/analysis';

const eventLabel = (t) => t === 'fire' ? 'FIRE' : t === 'smoke' ? 'SMOKE' : t === 'no_fire_smoke' ? 'NO FIRE' : t === 'uncertain' ? 'UNCERTAIN' : t === 'unavailable' ? 'MODEL NOT TRAINED' : '—';
const eventClass = (t) => t === 'fire' ? 'ev-fire' : t === 'smoke' ? 'ev-smoke' : t === 'no_fire_smoke' ? 'ev-nofire' : t === 'uncertain' ? 'ev-uncertain' : 'ev-untrained';
const pct = (v) => v === null || v === undefined ? 'Unavailable' : `${(v * 100).toFixed(v >= 0.1 ? 1 : 2)}%`;
const pct0 = (v) => v === null || v === undefined ? '—' : `${(v * 100).toFixed(0)}%`;
const num2 = (v) => v === null || v === undefined ? 'Unavailable' : Number(v).toFixed(2);

export default function App() {
  const [result, setResult] = useState(null);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState('');
  const [alerts, setAlerts] = useState([]);
  const [history, setHistory] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);
  const requestSeq = useRef(0);
  const lastAlertKey = useRef('');

  useEffect(() => () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  const refreshSide = () => {
    getAlerts().then((r) => {
      const items = Array.isArray(r.data) ? r.data : [];
      setAlerts(items.slice(0, 12).map((a, i) => ({
        id: a.id ?? `srv-${i}`,
        title: `${String(a.detection || 'event').toUpperCase()} — ${String(a.severity || 'Severity Unavailable')}`,
        detail: a.message || `model confidence unavailable`,
        severity: String(a.severity || 'Severity Unavailable'),
      })));
    }).catch(() => {});
    getHistory(20).then((r) => {
      setHistory(Array.isArray(r.data) ? r.data : []);
    }).catch(() => {});
    getModelInfo().then((r) => setModelInfo(r.data)).catch(() => {});
  };

  useEffect(() => { refreshSide(); }, []);

  const pushAlert = (norm) => {
    const det = norm.detection.type || 'unavailable';
    const sev = norm.severity.level || 'Severity Unavailable';
    if (det === 'no_fire_smoke' || det === 'unavailable' || det === 'uncertain') return;
    const key = `${det}|${sev}|${Math.round((norm.confidence || 0) * 50)}`;
    if (key === lastAlertKey.current) return;
    lastAlertKey.current = key;
    setAlerts((prev) => {
      if (prev.length && prev[0].title === `${det.toUpperCase()} — ${sev}`) return prev;
      return [{
        id: `local-${Date.now()}`,
        title: `${det.toUpperCase()} — ${sev}`,
        detail: `${pct0(norm.confidence)} confidence · AI-generated, unverified`,
        severity: sev,
      }, ...prev].slice(0, 12);
    });
  };

  const analyze = async (selectedFile) => {
    const seq = ++requestSeq.current;
    setAnalyzing(true);
    setError('');
    setResult(null);
    try {
      const response = await uploadImage(selectedFile);
      if (seq !== requestSeq.current) return; // stale response guard
      const norm = normalizeAnalysisResult(response.data);
      setResult(norm);
      pushAlert(norm);
      refreshSide();
    } catch (requestError) {
      if (seq !== requestSeq.current) return;
      const status = requestError.response?.status;
      const backendDetail = requestError.response?.data?.detail;
      const connectionDetail = requestError.request && !requestError.response
        ? 'Unable to reach the analysis backend at http://127.0.0.1:8001. Start it with: uvicorn backend.main:app --port 8001'
        : null;
      setError(status ? `Error ${status}: ${backendDetail || requestError.message}` : (backendDetail || connectionDetail || requestError.message || 'Analysis failed'));
    } finally {
      if (seq === requestSeq.current) setAnalyzing(false);
    }
  };

  const selectFile = (event) => {
    const selected = event.target.files[0] || null;
    event.target.value = '';
    if (!selected) return;
    setFile(selected);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(selected));
    analyze(selected);
  };

  const resetAll = () => {
    requestSeq.current += 1;
    setResult(null);
    setFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl('');
    setError('');
  };

  const r = result;
  const probs = r?.probabilities || {};
  const hasLocation = !!(r && r.latitude !== null && r.longitude !== null);
  const boxes = r?.boxes || [];
  const riskLevel = r?.overallRisk?.level || null;
  const riskScore = r?.overallRisk?.score;

  return (
    <main className="ff-dashboard">
      <header className="ff-topbar">
        <div>
          <h1>FOREST FIRE <span>AI</span></h1>
          <p>Upload an image — Fire / Smoke / No Fire — Confidence — Severity — Risk</p>
          <p className="ff-model-line">
            Model: {modelInfo ? `${modelInfo.architecture} — ${modelInfo.status}` : 'checking…'}
            {modelInfo?.growth ? ` · Temporal: ${modelInfo.growth.status}` : ''}
          </p>
        </div>
        <div className="ff-uploadbar">
          <label className="ff-upload-btn">Upload Image
            <input type="file" accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp" onChange={selectFile} />
          </label>
          {file && <span className="ff-filename">{file.name}</span>}
          {(result || file) && <button type="button" className="ff-reset-btn" onClick={resetAll}>Upload another</button>}
        </div>
      </header>

      {error && <section className="ff-card ff-error"><strong>Analysis failed</strong><span>{error}</span></section>}
      {analyzing && <section className="ff-card ff-loading"><span className="ff-spinner" />Analyzing image…</section>}
      {r && r.modelStatus === 'not_trained' && (
        <section className="ff-card ff-warn"><strong>MODEL NOT TRAINED</strong><span>No trained checkpoint loaded. Values below are unavailable — not predictions.</span></section>
      )}

      <div className="ff-layout">
        <section className="ff-main">
          <div className="ff-images">
            <div className="ff-card">
              <div className="ff-card-title">UPLOADED IMAGE</div>
              <div className="ff-media">
                {previewUrl ? <img src={previewUrl} alt="Uploaded" /> : <div className="ff-empty">No image uploaded yet</div>}
              </div>
            </div>
            <div className="ff-card">
              <div className="ff-card-title">AI ANALYSIS RESULT</div>
              <div className="ff-media">
                {previewUrl ? (
                  <div className="ff-overlay-wrap">
                    <img src={previewUrl} alt="Analysis result" />
                    {boxes.map((b, i) => (
                      <div key={i} className="ff-box" style={{
                        left: `${(b.x || 0) * 100}%`, top: `${(b.y || 0) * 100}%`,
                        width: `${(b.width || 0) * 100}%`, height: `${(b.height || 0) * 100}%`,
                      }}>
                        <span>{b.label} {pct0(b.confidence)}</span>
                      </div>
                    ))}
                  </div>
                ) : <div className="ff-empty">Result appears after upload</div>}
              </div>
              {r && <div className="ff-model-note">{r.modelUsed || ''}{r.evidence?.length ? ` · ${r.evidence[0]}` : ''}</div>}
            </div>
          </div>

          <div className="ff-cards">
            <div className={`ff-card ff-metric ${eventClass(r?.detection.type)}`}>
              <div className="ff-card-title">EVENT DETECTION</div>
              <div className="ff-big">{r ? eventLabel(r.detection.type) : '—'}</div>
              <div className="ff-sub">{r ? pct(r.confidence) : 'Upload to analyze'}</div>
              {r && <div className="ff-probs">
                <span>Fire: {pct(probs.fire)}</span>
                <span>Smoke: {pct(probs.smoke)}</span>
                <span>No Fire: {pct(probs.no_fire)}</span>
              </div>}
            </div>
            <div className="ff-card ff-metric">
              <div className="ff-card-title">SEVERITY</div>
              <div className="ff-big">{r ? (r.severity.level || 'Severity Unavailable') : '—'}</div>
              <div className="ff-sub">{r ? (r.severity.score !== null && r.severity.score !== undefined ? `score ${num2(r.severity.score)}` : 'Severity Unavailable') : 'Upload to analyze'}</div>
              {r?.severity.method && <div className="ff-note">{r.severity.method}</div>}
            </div>
            <div className="ff-card ff-metric">
              <div className="ff-card-title">WILDFIRE RISK (0–1)</div>
              <div className="ff-big">{r ? (riskLevel || 'Risk Unavailable') : '—'}</div>
              <div className="ff-sub">{r ? (riskScore !== null && riskScore !== undefined ? num2(riskScore) : 'Risk Unavailable') : 'Upload to analyze'}</div>
              {r?.overallRisk.method && <div className="ff-note">{r.overallRisk.method}</div>}
            </div>
            <div className="ff-card ff-metric">
              <div className="ff-card-title">GROWTH PREDICTION</div>
              <div className="ff-big dim">{r ? (r.growth.status || 'INSUFFICIENT_TEMPORAL_DATA') : '—'}</div>
              <div className="ff-sub">{r ? (r.growth.message || 'Single image; temporal sequence required.') : 'Upload to analyze'}</div>
            </div>
          </div>

          <div className="ff-card ff-chart">
            <div className="ff-card-title">GROWTH / RISK CHART</div>
            {!r ? <div className="ff-empty">Upload an image to see analysis</div> : (
              <>
                <div className="ff-empty">Temporal data unavailable — temporal forecast needs ≥2 ordered frames. Below: live class probabilities (not an LSTM curve).</div>
                <div className="ff-bars">
                  <ProbBar name="Fire" value={probs.fire} />
                  <ProbBar name="Smoke" value={probs.smoke} />
                  <ProbBar name="No Fire" value={probs.no_fire} />
                </div>
              </>
            )}
            {history.length > 0 && (
              <>
                <div className="ff-card-title" style={{ marginTop: 12 }}>DETECTION HISTORY (RISK SCORE)</div>
                <div className="ff-history-chart">
                  {history.slice(0, 12).reverse().map((h) => (
                    <div key={h.id} className="ff-hbar" title={`#${h.id} ${h.detection} risk ${h.risk_score ?? h.confidence ?? 'n/a'}`}>
                      <div style={{ height: `${Math.round(((h.risk_score ?? h.confidence ?? 0)) * 100)}%` }} />
                      <span>{h.id}</span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>

          <div className="ff-card">
            <div className="ff-card-title">DETECTION HISTORY</div>
            {history.length === 0
              ? <div className="ff-empty">No detections saved yet</div>
              : (
                <table className="ff-table">
                  <thead><tr><th>ID</th><th>Time</th><th>Event</th><th>Conf</th><th>Severity</th><th>Risk</th></tr></thead>
                  <tbody>
                    {history.slice(0, 12).map((h) => (
                      <tr key={h.id}>
                        <td>{h.id}</td>
                        <td>{h.timestamp ? String(h.timestamp).slice(0, 19).replace('T', ' ') : '—'}</td>
                        <td>{String(h.detection || '').toUpperCase()}</td>
                        <td>{h.confidence !== null && h.confidence !== undefined ? pct0(h.confidence) : '—'}</td>
                        <td>{h.severity || '—'}</td>
                        <td>{h.risk_score !== null && h.risk_score !== undefined ? num2(h.risk_score) : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
          </div>
        </section>

        <aside className="ff-side">
          <div className="ff-card">
            <div className="ff-card-title">LOCATION MAP</div>
            <div className="ff-map">
              <MapContainer center={hasLocation ? [r.latitude, r.longitude] : [20, 0]} zoom={hasLocation ? 6 : 2} scrollWheelZoom={false} className="ff-leaflet">
                <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                {hasLocation && (
                  <CircleMarker center={[r.latitude, r.longitude]} radius={10}>
                    <Popup>{eventLabel(r.detection.type)} · {pct(r.confidence)}</Popup>
                  </CircleMarker>
                )}
              </MapContainer>
              {!hasLocation && <div className="ff-map-badge">LOCATION UNAVAILABLE — {r ? r.locationStatus : 'upload with EXIF GPS or coordinates'}</div>}
            </div>
            {r && <div className="ff-sub" style={{ marginTop: 8 }}>Coords: {hasLocation ? `${r.latitude}, ${r.longitude}` : 'Unavailable'} · Affected area: {r.affectedArea.value !== null ? `${r.affectedArea.value}% image-space` : 'NOT AVAILABLE (no segmentation model)'}</div>}
          </div>

          <div className="ff-card">
            <div className="ff-card-title">EARLY-WARNING ALERT</div>
            {!r ? <div className="ff-sub">Upload an image to check</div> : (
              <>
                <div className={`ff-big ${r.alert.level.includes('HIGH') ? 'alert-on' : ''}`}>{r.alert.level}</div>
                <div className="ff-sub">AI-generated, unverified until confirmed{hasLocation ? ` · ${r.latitude}, ${r.longitude}` : ' · location unavailable'}</div>
              </>
            )}
          </div>

          <div className="ff-card">
            <div className="ff-card-title">SATELLITE EARLY WARNING</div>
            {!r ? <div className="ff-sub">Upload an image to check</div> : (
              <>
                <div className="ff-sub">Status: {r.satellite.available ? 'Available' : 'Unavailable'}</div>
                {hasLocation ? (
                  r.satellite.available
                    ? <div className="ff-sub">{r.satellite.risk} signal · score {r.satellite.riskScore ?? '—'} · {r.satellite.source}</div>
                    : <div className="ff-sub">{r.satellite.status}</div>
                ) : <div className="ff-sub">Location required for satellite analysis</div>}
              </>
            )}
            <div className="ff-note">Seasonal / province-level MODIS signal only.</div>
          </div>

          <div className="ff-card">
            <div className="ff-card-title">ALERTS</div>
            {alerts.length === 0
              ? <div className="ff-empty">No alerts yet</div>
              : <div className="ff-alerts">{alerts.map((a) => (
                <div key={a.id} className={`ff-alert sev-${a.severity}`}>
                  <b>{a.title}</b><span>{a.detail}</span>
                </div>
              ))}</div>}
          </div>
        </aside>
      </div>
    </main>
  );
}

function ProbBar({ name, value }) {
  const v = value === null || value === undefined ? 0 : Math.max(0, Math.min(1, value));
  return (
    <div className="ff-bar-row">
      <span>{name}</span>
      <div className="ff-bar"><div style={{ width: `${v * 100}%` }} /></div>
      <b>{value === null || value === undefined ? 'Unavailable' : `${(value * 100).toFixed(1)}%`}</b>
    </div>
  );
}
