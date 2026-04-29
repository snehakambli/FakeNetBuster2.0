import React, { useEffect, useState } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { ArrowLeft, Download, AlertTriangle, CheckCircle, Clock, Cpu } from 'lucide-react'
import { getReport } from '../services/api'
import AnalysisViewer from '../components/AnalysisViewer'

export default function Report() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [report, setReport]   = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState(null)

  useEffect(() => {
    if (!id) return
    getReport(id)
      .then(setReport)
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [id])

  const handleDownload = () => {
    if (!report) return
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href     = url
    a.download = `fakenetbuster_report_${report.report_id?.slice(0, 8)}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (loading) {
    return (
      <div className="min-h-screen cyber-grid pt-24 flex items-center justify-center">
        <div className="spinner" />
      </div>
    )
  }

  if (error || !report) {
    return (
      <div className="min-h-screen cyber-grid pt-24 px-4">
        <div className="max-w-xl mx-auto text-center py-24">
          <div className="w-14 h-14 rounded-2xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center mx-auto mb-5">
            <AlertTriangle size={24} className="text-rose-400" />
          </div>
          <h2 className="text-xl font-bold text-white mb-2">Report Not Found</h2>
          <p className="text-slate-500 text-sm mb-7">{error || 'This report does not exist.'}</p>
          <button onClick={() => navigate(-1)} className="btn-primary inline-flex items-center gap-2">
            <ArrowLeft size={15} /> Go Back
          </button>
        </div>
      </div>
    )
  }

  const isFake = report.prediction === 'fake'

  return (
    <div className="min-h-screen cyber-grid pt-24 pb-16 px-4">
      <div className="max-w-4xl mx-auto">

        {/* Top bar */}
        <div className="flex items-center justify-between mb-6">
          <button onClick={() => navigate(-1)} className="flex items-center gap-1.5 text-slate-500 hover:text-white text-sm transition-colors">
            <ArrowLeft size={14} /> Back
          </button>
          <button
            onClick={handleDownload}
            className="flex items-center gap-2 px-4 py-2 rounded-xl border border-[#1a2540] text-slate-500 hover:text-white hover:border-[#243050] text-sm transition-all"
          >
            <Download size={13} /> Export JSON
          </button>
        </div>

        {/* Report header */}
        <div className="cyber-card p-6 mb-5">
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${isFake ? 'bg-rose-500/10 border border-rose-500/20' : 'bg-emerald-500/10 border border-emerald-500/20'}`}>
                {isFake
                  ? <AlertTriangle size={20} className="text-rose-400" />
                  : <CheckCircle  size={20} className="text-emerald-400" />
                }
              </div>
              <div>
                <h1 className="text-xl font-bold text-white capitalize">{report.content_type} Analysis</h1>
                <p className="text-slate-600 text-xs mt-0.5 font-mono">ID: {report.report_id}</p>
              </div>
            </div>
            <span className={`tag-pill ${isFake ? 'border-rose-500/30 text-rose-400 bg-rose-500/8' : 'border-emerald-500/30 text-emerald-400 bg-emerald-500/8'}`}>
              {report.prediction?.toUpperCase()}
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-5">
            {[
              { label: 'Confidence',  value: `${Math.round(report.confidence_score * 100)}%` },
              { label: 'Risk Level',  value: report.risk_level },
              { label: 'Date',        value: new Date(report.timestamp).toLocaleDateString() },
              { label: 'Processing',  value: `${report.processing_time_sec}s` },
            ].map(({ label, value }) => (
              <div key={label} className="bg-[#060912] rounded-xl p-3 border border-[#1a2540]">
                <p className="label-xs mb-1">{label}</p>
                <p className="text-white font-semibold text-sm">{value}</p>
              </div>
            ))}
          </div>
        </div>

        <AnalysisViewer report={report} />

        {/* Raw JSON */}
        <details className="mt-5 cyber-card overflow-hidden">
          <summary className="px-5 py-3.5 cursor-pointer text-slate-500 hover:text-white text-sm font-medium select-none transition-colors">
            Raw JSON
          </summary>
          <pre className="px-5 pb-5 pt-3 text-xs text-slate-500 overflow-auto max-h-80 border-t border-[#1a2540] font-mono leading-relaxed">
            {JSON.stringify(report, null, 2)}
          </pre>
        </details>
      </div>
    </div>
  )
}
