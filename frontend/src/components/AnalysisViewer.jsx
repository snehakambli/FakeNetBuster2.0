import React from 'react'
import { AlertTriangle, CheckCircle, Clock, Cpu, Zap, ExternalLink, ShieldCheck, Newspaper, Bot } from 'lucide-react'

const RiskBadge = ({ level }) => {
  const cls = {
    CRITICAL: 'risk-critical',
    HIGH:     'risk-high',
    MEDIUM:   'risk-medium',
    LOW:      'risk-low',
  }
  return (
    <span className={`tag-pill ${cls[level] || cls.LOW}`}>{level}</span>
  )
}

const ConfidenceGauge = ({ value, prediction }) => {
  const isFake = prediction === 'fake'
  const color  = isFake ? '#f87171' : '#34d399'
  const track  = isFake ? 'rgba(248,113,113,0.1)' : 'rgba(52,211,153,0.1)'
  const pct    = Math.round((value ?? 0) * 100)
  const size   = 152
  const sw     = 10
  const r      = (size - sw) / 2
  const circ   = 2 * Math.PI * r
  const dash   = (pct / 100) * circ

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}
          className={isFake ? 'gauge-glow-fake' : 'gauge-glow-real'}
        >
          <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={track} strokeWidth={sw} />
          <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={color} strokeWidth={sw}
            strokeLinecap="round" strokeDasharray={`${dash} ${circ - dash}`}
            style={{ transition: 'stroke-dasharray 0.7s cubic-bezier(0.4,0,0.2,1)' }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-3xl font-extrabold leading-none tracking-tight" style={{ color }}>{pct}%</span>
          <span className="text-xs text-slate-500 mt-0.5">confidence</span>
        </div>
      </div>
      <span className={`text-sm font-bold uppercase tracking-widest ${isFake ? 'text-rose-400' : 'text-emerald-400'}`}>
        {prediction ?? 'unknown'}
      </span>
    </div>
  )
}

export default function AnalysisViewer({ report }) {
  if (!report) return null

  const {
    prediction, confidence_score, risk_level, content_type,
    detected_anomalies, possible_generation_method, model_used,
    analysis_details, summary, processing_time_sec,
    signals, related_links, inline_images,
  } = report

  return (
    <div className="fade-in space-y-4">

      {/* Header card */}
      <div className={`cyber-card p-5 border-l-[3px] ${prediction === 'fake' ? 'border-l-rose-500' : 'border-l-emerald-500'}`}>
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            {prediction === 'fake'
              ? <AlertTriangle size={20} className="text-rose-400 shrink-0" />
              : <CheckCircle  size={20} className="text-emerald-400 shrink-0" />
            }
            <div>
              <h2 className="text-base font-bold text-white">Analysis Complete</h2>
              <p className="text-slate-500 text-xs capitalize mt-0.5">{content_type} · {model_used}</p>
            </div>
          </div>
          <div className="flex items-center gap-2.5">
            <RiskBadge level={risk_level} />
            <span className="flex items-center gap-1 text-xs text-slate-600 font-mono">
              <Clock size={11} />{processing_time_sec}s
            </span>
          </div>
        </div>

        {summary && (
          <p className="mt-4 text-slate-400 text-sm leading-relaxed border-l-2 border-sky-400/25 pl-3">
            {report.analysis_details?.ai_verdict?.available
              ? report.analysis_details.ai_verdict.explanation
              : (report.result_statement || summary)}
          </p>
        )}
      </div>

      {/* Visualizations */}
      {inline_images && Object.keys(inline_images).length > 0 && (
        <VisualizationPanel images={inline_images} prediction={prediction} />
      )}

      {/* Gauge + details */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="cyber-card p-6 flex flex-col items-center justify-center gap-4"
          style={{
            background: prediction === 'fake'
              ? 'linear-gradient(160deg, #0d1225 60%, rgba(248,113,113,0.05))'
              : 'linear-gradient(160deg, #0d1225 60%, rgba(52,211,153,0.05))',
          }}
        >
          <ConfidenceGauge value={confidence_score} prediction={prediction} />
          <div className="w-full border-t border-[#1a2540] pt-3 flex justify-between items-center text-xs text-slate-600">
            <span>Risk level</span>
            <RiskBadge level={risk_level} />
          </div>
        </div>

        <div className="md:col-span-2 cyber-card p-5 space-y-4">
          <div>
            <p className="label-xs mb-1.5">Possible Generation Method</p>
            <div className="flex items-center gap-2">
              <Cpu size={13} className="text-sky-400" />
              <span className="text-white text-sm">{possible_generation_method}</span>
            </div>
          </div>

          {detected_anomalies?.length > 0 && (
            <div>
              <p className="label-xs mb-2">Detected Anomalies</p>
              <div className="space-y-1.5">
                {detected_anomalies.map((a, i) => (
                  <div key={i} className="flex items-start gap-2 text-sm">
                    <Zap size={11} className="text-amber-400 mt-0.5 shrink-0" />
                    <span className="text-slate-300">{a}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Content-specific details */}
      {analysis_details && <AnalysisDetails type={content_type} details={analysis_details} />}

      {/* AI Verdict */}
      {content_type === 'news' && report.analysis_details?.ai_verdict?.available && (
        <AiVerdictCard ai={report.analysis_details.ai_verdict} />
      )}

      {/* Signals */}
      {content_type === 'news' && signals?.length > 0 && (
        <div className="cyber-card p-5">
          <h3 className="label-xs flex items-center gap-2 mb-4">
            <ShieldCheck size={13} className="text-sky-400" />
            Fact-Check Signal Breakdown
          </h3>
          <div className="space-y-2">
            {signals.map((s, i) => (
              <div key={i} className="flex items-start gap-2 text-sm">
                <span className="text-sky-400 mt-0.5 shrink-0">›</span>
                <span className="text-slate-400">{s}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Related links */}
      {content_type === 'news' && related_links?.length > 0 && (
        <div className="cyber-card p-5">
          <h3 className="label-xs flex items-center gap-2 mb-4">
            <Newspaper size={13} className="text-amber-400" />
            Related Links &amp; Fact Checks
          </h3>
          <div className="space-y-2">
            {related_links.filter(l => l.url && l.url !== 'https://groq.com').slice(0, 8).map((link, i) => (
              <a key={i} href={link.url} target="_blank" rel="noopener noreferrer"
                className="flex items-start gap-3 p-3 rounded-xl bg-[#060912] hover:bg-[#0b0f1e] border border-[#1a2540] hover:border-sky-400/25 transition-all group"
              >
                <ExternalLink size={13} className="text-sky-400 mt-0.5 shrink-0 group-hover:text-sky-300" />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-white text-sm font-medium line-clamp-1">{link.title || link.domain}</span>
                    {link.is_fact_check && (
                      <span className="tag-pill border-violet-500/30 text-violet-300 bg-violet-500/10 shrink-0">Fact Check</span>
                    )}
                    {link.is_credible && !link.is_fact_check && (
                      <span className="tag-pill border-emerald-500/30 text-emerald-300 bg-emerald-500/10 shrink-0">Credible</span>
                    )}
                  </div>
                  {link.snippet && <p className="text-slate-600 text-xs mt-1 line-clamp-2">{link.snippet}</p>}
                  <p className="text-slate-700 text-xs mt-0.5 font-mono">{link.domain}</p>
                </div>
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function AiVerdictCard({ ai }) {
  const isFake = ai.verdict === 'fake'
  const pct    = Math.round((ai.confidence || 0) * 100)

  return (
    <div className={`cyber-card p-5 border-l-[3px] ${isFake ? 'border-l-rose-500' : 'border-l-emerald-500'}`}>
      <div className="flex items-center gap-2 mb-3">
        <Bot size={14} className="text-sky-400" />
        <span className="label-xs">{ai.source || 'AI Analysis'} Verdict</span>
        <span className={`ml-auto tag-pill ${isFake ? 'border-rose-500/30 text-rose-400 bg-rose-500/8' : 'border-emerald-500/30 text-emerald-400 bg-emerald-500/8'}`}>
          {ai.verdict?.toUpperCase()} · {pct}%
        </span>
      </div>
      {ai.explanation && <p className="text-slate-300 text-sm leading-relaxed mb-3">{ai.explanation}</p>}
      {isFake && ai.red_flags?.length > 0 && (
        <div className="space-y-1.5">
          {ai.red_flags.map((flag, i) => (
            <div key={i} className="flex items-start gap-2 text-sm">
              <Zap size={11} className="text-rose-400 mt-0.5 shrink-0" />
              <span className="text-slate-400">{flag}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function VisualizationPanel({ images, prediction }) {
  const [active, setActive] = React.useState(Object.keys(images)[0])
  const entries = Object.entries(images)

  return (
    <div className="cyber-card p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="label-xs">Visual Analysis</h3>
        {prediction === 'fake' && (
          <span className="text-xs text-rose-400">red boxes mark flagged regions</span>
        )}
      </div>

      {entries.length > 1 && (
        <div className="flex gap-2 flex-wrap">
          {entries.map(([label]) => (
            <button key={label} onClick={() => setActive(label)}
              className={`px-3 py-1 rounded-lg text-xs font-medium border transition-colors ${
                active === label
                  ? 'bg-sky-400/10 border-sky-400/35 text-sky-300'
                  : 'border-[#1a2540] text-slate-500 hover:text-slate-300'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      )}

      <div className="relative rounded-xl overflow-hidden border border-[#1a2540] bg-[#060912]">
        <img src={images[active]} alt={active} className="w-full object-contain max-h-[500px]" />
        <div className="absolute bottom-0 left-0 right-0 px-3 py-2 bg-gradient-to-t from-black/60 to-transparent">
          <p className="text-xs text-slate-500 font-mono">{active}</p>
        </div>
      </div>
    </div>
  )
}

function MetricBox({ label, value }) {
  return (
    <div className="bg-[#060912] rounded-xl p-3 border border-[#1a2540]">
      <p className="label-xs mb-1">{label}</p>
      <p className="text-white font-semibold text-sm">{value}</p>
    </div>
  )
}

function ScoreBar({ label, value, color = 'bg-sky-400' }) {
  const pct = Math.round((value ?? 0) * 100)
  return (
    <div className="bg-[#060912] rounded-xl p-3 border border-[#1a2540]">
      <div className="flex justify-between text-xs mb-1.5">
        <span className="text-slate-500">{label}</span>
        <span className="text-white font-mono">{pct}%</span>
      </div>
      <div className="w-full bg-[#1a2540] rounded-full h-1.5">
        <div className={`h-1.5 rounded-full ${color} transition-all duration-700`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

function AnalysisDetails({ type, details }) {
  if (!details) return null

  return (
    <div className="cyber-card p-5">
      <h3 className="label-xs mb-4">Detailed Analysis</h3>

      {type === 'video' && (
        <div className="space-y-3">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <MetricBox label="Duration"          value={`${details.video_duration_sec}s`} />
            <MetricBox label="FPS"               value={details.fps} />
            <MetricBox label="Frames Analyzed"   value={details.total_frames_analyzed} />
            <MetricBox label="Suspicious Frames" value={details.suspicious_frames?.length || 0} />
          </div>
          {details.suspicious_timestamps_sec?.length > 0 && (
            <div>
              <p className="label-xs mb-2">Suspicious Timestamps</p>
              <div className="flex flex-wrap gap-1.5">
                {details.suspicious_timestamps_sec.slice(0, 15).map((t, i) => (
                  <span key={i} className="px-2 py-0.5 bg-rose-500/8 text-rose-400 text-xs rounded-lg border border-rose-500/20 font-mono">{t}s</span>
                ))}
              </div>
            </div>
          )}
          {details.suspicious_frames_b64?.length > 0 && (
            <SuspiciousFramesPanel frames={details.suspicious_frames_b64} />
          )}
        </div>
      )}

      {type === 'audio' && (
        <div className="space-y-3">
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <MetricBox label="Pitch Anomaly Score"     value={details.pitch_anomaly_score?.toFixed(3) ?? '—'} />
            <MetricBox label="Spectral Inconsistency"  value={details.spectral_inconsistency_score?.toFixed(3) ?? '—'} />
            <MetricBox label="Anomalous Frames"        value={details.anomalous_time_frames?.length ?? 0} />
          </div>
          {details.pitch_anomaly_score != null && (
            <ScoreBar label="Pitch Anomaly" value={details.pitch_anomaly_score} color="bg-amber-400" />
          )}
          {details.spectral_inconsistency_score != null && (
            <ScoreBar label="Spectral Inconsistency" value={details.spectral_inconsistency_score} color="bg-orange-400" />
          )}
          {details.anomalous_time_frames?.length > 0 && (
            <div>
              <p className="label-xs mb-2">Anomalous Time Frames</p>
              <div className="flex flex-wrap gap-1.5">
                {details.anomalous_time_frames.slice(0, 20).map((f, i) => (
                  <span key={i} className="px-2 py-0.5 bg-rose-500/8 text-rose-400 text-xs rounded-lg border border-rose-500/20 font-mono">{f}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {type === 'image' && (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          <MetricBox label="Noise Inconsistency"  value={details.noise_inconsistency_score?.toFixed(3)} />
          <MetricBox label="Color Inconsistency"  value={details.color_inconsistency_score?.toFixed(3)} />
          <MetricBox label="GradCAM Available"    value={details.gradcam_available ? 'Yes' : 'No'} />
        </div>
      )}

      {type === 'news' && (
        <div className="space-y-3">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <MetricBox label="Word Count"           value={details.word_count} />
            <MetricBox label="Clickbait Indicators" value={details.text_features?.clickbait_indicators ?? '—'} />
            <MetricBox label="Credibility Signals"  value={details.text_features?.credibility_indicators ?? '—'} />
            <MetricBox label="Cross-References"     value={details.cross_reference_count ?? 0} />
          </div>
          {details.fact_check_apis && (
            <div className="flex flex-wrap gap-2">
              {[
                { key: 'google_factcheck_available', label: 'Google Fact Check' },
                { key: 'claimbuster_available',      label: 'ClaimBuster' },
                { key: 'newsapi_available',          label: 'NewsAPI' },
              ].map(({ key, label }) => (
                <span key={key} className={`tag-pill ${
                  details.fact_check_apis[key]
                    ? 'border-emerald-500/30 text-emerald-400 bg-emerald-500/8'
                    : 'border-[#1a2540] text-slate-600 bg-transparent'
                }`}>
                  {label}: {details.fact_check_apis[key] ? 'active' : 'off'}
                </span>
              ))}
            </div>
          )}
          {details.claimbuster_score != null && (
            <ScoreBar label="ClaimBuster Check-Worthy Score" value={details.claimbuster_score} color="bg-sky-400" />
          )}
          {details.text_preview && (
            <div className="bg-[#060912] rounded-xl p-3 border border-[#1a2540]">
              <p className="label-xs mb-1.5">Text Preview</p>
              <p className="text-slate-400 text-sm leading-relaxed">{details.text_preview}</p>
            </div>
          )}
        </div>
      )}

      {type === 'document' && (
        <div className="grid grid-cols-2 gap-3">
          <MetricBox label="ELA Score"           value={details.ela_score?.toFixed(3)} />
          <MetricBox label="Font Inconsistency"  value={details.font_inconsistency_score?.toFixed(3)} />
          <MetricBox label="Seal Regions"        value={details.seal_regions_detected} />
          <MetricBox label="OCR Extracted"       value={details.ocr_text_extracted ? 'Yes' : 'No'} />
        </div>
      )}
    </div>
  )
}

function SuspiciousFramesPanel({ frames }) {
  return (
    <div className="space-y-2">
      <p className="label-xs flex items-center gap-1.5">
        <span className="w-1.5 h-1.5 rounded-full bg-rose-400 inline-block" />
        Flagged Frames
      </p>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {frames.map((f, i) => (
          <div key={i} className="rounded-xl overflow-hidden border border-rose-500/25 bg-[#060912]">
            <img src={f.data} alt={`Frame ${f.frame}`} className="w-full object-cover" />
            <div className="px-2 py-1.5 flex items-center justify-between">
              <span className="text-xs text-rose-400 font-mono">#{f.frame}</span>
              <span className="text-xs text-slate-700">{i + 1}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
