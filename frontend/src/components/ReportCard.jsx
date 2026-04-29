import React from 'react'
import { Link } from 'react-router-dom'
import { AlertTriangle, CheckCircle, FileImage, FileVideo, FileAudio, FileText, Newspaper, Clock, ArrowRight } from 'lucide-react'

const typeIcons = {
  image:    FileImage,
  video:    FileVideo,
  audio:    FileAudio,
  document: FileText,
  news:     Newspaper,
}

const riskStyle = {
  CRITICAL: 'border-rose-500/30 text-rose-400 bg-rose-500/8',
  HIGH:     'border-orange-500/30 text-orange-400 bg-orange-500/8',
  MEDIUM:   'border-amber-500/30 text-amber-400 bg-amber-500/8',
  LOW:      'border-emerald-500/30 text-emerald-400 bg-emerald-500/8',
}

export default function ReportCard({ report, onDelete }) {
  const { report_id, timestamp, content_type, prediction, confidence_score, risk_level } = report
  const TypeIcon = typeIcons[content_type] || FileText
  const isFake   = prediction === 'fake'

  return (
    <div className="cyber-card p-4 hover:border-[#243050] transition-all group">
      <div className="flex items-center gap-4">
        {/* Icon */}
        <div className="w-10 h-10 rounded-xl bg-[#060912] border border-[#1a2540] flex items-center justify-center shrink-0">
          <TypeIcon size={16} className="text-slate-500" />
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-white font-medium text-sm capitalize">{content_type}</span>
            <span className={`tag-pill ${riskStyle[risk_level] || riskStyle.LOW}`}>{risk_level}</span>
            {isFake
              ? <span className="flex items-center gap-1 text-rose-400 text-xs"><AlertTriangle size={10} />FAKE</span>
              : <span className="flex items-center gap-1 text-emerald-400 text-xs"><CheckCircle size={10} />REAL</span>
            }
          </div>
          <div className="flex items-center gap-3 mt-1 text-xs text-slate-600 font-mono">
            <span className="flex items-center gap-1"><Clock size={10} />{new Date(timestamp).toLocaleString()}</span>
            <span>{Math.round(confidence_score * 100)}%</span>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2 shrink-0">
          <Link
            to={`/report/${report_id}`}
            className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-sky-400/8 text-sky-400 text-xs hover:bg-sky-400/15 border border-sky-400/20 transition-colors"
          >
            View <ArrowRight size={11} />
          </Link>
          {onDelete && (
            <button
              onClick={() => onDelete(report_id)}
              className="w-7 h-7 flex items-center justify-center rounded-lg text-slate-600 hover:text-rose-400 hover:bg-rose-400/10 text-xs transition-colors"
            >
              ✕
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
