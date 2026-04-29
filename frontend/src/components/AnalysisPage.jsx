import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import toast from 'react-hot-toast'
import { Loader, ArrowRight, RotateCcw, Sparkles } from 'lucide-react'
import UploadBox from './UploadBox'
import AnalysisViewer from './AnalysisViewer'
import { uploadFile, analyzeFile, analyzeNews, getPreviewUrl } from '../services/api'

export default function AnalysisPage({
  title, subtitle, icon: Icon,
  accentColor, borderColor, bgColor,
  acceptedTypes, acceptedExts, contentTypeHint,
  isTextInput, textPlaceholder, tips,
}) {
  const navigate = useNavigate()
  const [file, setFile]                       = useState(null)
  const [previewUrl, setPreviewUrl]           = useState(null)
  const [transcodedUrl, setTranscodedUrl]     = useState(null)
  const [videoPreviewError, setVideoPreviewError] = useState(false)
  const prevUrlRef      = useRef(null)
  const uploadedPathRef = useRef(null)
  const [textInput, setTextInput]             = useState('')
  const [uploading, setUploading]             = useState(false)
  const [analyzing, setAnalyzing]             = useState(false)
  const [uploadProgress, setUploadProgress]   = useState(0)
  const [report, setReport]                   = useState(null)
  const [step, setStep]                       = useState('idle')

  const isVideo    = contentTypeHint === 'video'
  const isImage    = contentTypeHint === 'image'
  const isAudio    = contentTypeHint === 'audio'
  const isDocument = contentTypeHint === 'document'
  const canSubmit  = isTextInput ? textInput.trim().length > 0 : !!file
  const busy       = uploading || analyzing

  const handleFileSelect = (f) => {
    if (prevUrlRef.current) { URL.revokeObjectURL(prevUrlRef.current); prevUrlRef.current = null }
    setFile(f)
    setTranscodedUrl(null)
    setVideoPreviewError(false)
    uploadedPathRef.current = null
    if (f) {
      const url = URL.createObjectURL(f)
      prevUrlRef.current = url
      setPreviewUrl(url)
      if (isVideo) {
        uploadFile(f, () => {})
          .then(uploaded => { uploadedPathRef.current = uploaded.file_path; setTranscodedUrl(getPreviewUrl(uploaded.file_path)) })
          .catch(() => { setVideoPreviewError(true); toast.error('Video preview failed — file will still be analyzed') })
      }
    } else {
      setPreviewUrl(null)
    }
  }

  const reset = () => {
    handleFileSelect(null)
    setTextInput('')
    setReport(null)
    setStep('idle')
    setUploadProgress(0)
    setTranscodedUrl(null)
    setVideoPreviewError(false)
    uploadedPathRef.current = null
  }

  const handleAnalyze = async () => {
    if (!canSubmit) return
    try {
      if (isTextInput) {
        setAnalyzing(true); setStep('analyzing')
        const isUrl = textInput.startsWith('http')
        const result = await analyzeNews(isUrl ? null : textInput, isUrl ? textInput : null)
        setReport(result); setStep('done')
        toast.success('Analysis complete')
      } else {
        setUploading(true); setStep('uploading')
        const uploaded = uploadedPathRef.current
          ? { file_path: uploadedPathRef.current }
          : await uploadFile(file, setUploadProgress)
        setUploading(false); setAnalyzing(true); setStep('analyzing')
        const result = await analyzeFile(uploaded.file_path, contentTypeHint)
        setReport(result); setStep('done')
        toast.success('Analysis complete')
      }
    } catch (err) {
      toast.error(err.message || 'Analysis failed')
      setStep('idle')
    } finally {
      setUploading(false); setAnalyzing(false)
    }
  }

  return (
    <div className="min-h-screen cyber-grid pt-24 pb-16 px-4">
      <div className="max-w-3xl mx-auto">

        {/* Page header */}
        <div className="flex items-center gap-4 mb-8">
          <div className={`w-12 h-12 rounded-xl ${bgColor} border ${borderColor} flex items-center justify-center shrink-0`}>
            <Icon size={22} className={accentColor} />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">{title}</h1>
            <p className="text-slate-500 text-sm mt-0.5">{subtitle}</p>
          </div>
        </div>

        {/* Upload / input */}
        {!report && (
          <div className="space-y-4 fade-in">
            <div className="cyber-card p-5">
              {isTextInput ? (
                <div className="space-y-3">
                  <label className="label-xs">Article text or URL</label>
                  <textarea
                    value={textInput}
                    onChange={e => setTextInput(e.target.value)}
                    placeholder={textPlaceholder}
                    rows={7}
                    disabled={busy}
                    className="cyber-input"
                  />
                  <p className="text-slate-600 text-xs">Paste raw text or a full article URL (https://...)</p>
                </div>
              ) : (
                <UploadBox
                  onFileSelect={handleFileSelect}
                  uploading={uploading}
                  uploadProgress={uploadProgress}
                  acceptedTypes={acceptedTypes}
                  acceptedExts={acceptedExts}
                />
              )}
            </div>

            {/* Image preview */}
            {previewUrl && isImage && (
              <div className="cyber-card p-4">
                <p className="label-xs mb-3">Preview</p>
                <div className="rounded-xl overflow-hidden border border-[#1a2540] bg-[#060912]">
                  <img src={previewUrl} alt="preview" className="w-full object-contain max-h-72" />
                </div>
                <p className="text-slate-600 text-xs mt-2 font-mono">{file.name} · {(file.size / 1024).toFixed(1)} KB</p>
              </div>
            )}

            {/* Video preview */}
            {previewUrl && isVideo && (
              <div className="cyber-card p-4">
                <p className="label-xs mb-3">Preview</p>
                {transcodedUrl ? (
                  <video key={transcodedUrl} controls className="w-full rounded-xl" style={{ maxHeight: 320 }} src={transcodedUrl}>
                    Your browser does not support video playback.
                  </video>
                ) : videoPreviewError ? (
                  <div className="flex items-center justify-center bg-[#060912] rounded-xl py-8 border border-[#1a2540]">
                    <p className="text-slate-600 text-sm">Preview unavailable — file is ready for analysis</p>
                  </div>
                ) : (
                  <div className="flex items-center justify-center gap-3 bg-[#060912] rounded-xl py-8 border border-[#1a2540]">
                    <div className="w-4 h-4 border-2 border-violet-400 border-t-transparent rounded-full animate-spin" />
                    <p className="text-slate-500 text-sm">Preparing preview...</p>
                  </div>
                )}
                <p className="text-slate-600 text-xs mt-2 font-mono">{file.name} · {(file.size / (1024 * 1024)).toFixed(1)} MB</p>
              </div>
            )}

            {/* Audio preview */}
            {previewUrl && isAudio && (
              <div className="cyber-card p-4">
                <p className="label-xs mb-3">Preview</p>
                <div className="rounded-xl border border-[#1a2540] bg-[#060912] px-4 py-3">
                  <audio key={previewUrl} controls className="w-full" style={{ accentColor: '#38bdf8' }}>
                    <source src={previewUrl} type={file.type || 'audio/wav'} />
                  </audio>
                </div>
                <p className="text-slate-600 text-xs mt-2 font-mono">{file.name} · {(file.size / 1024).toFixed(1)} KB</p>
              </div>
            )}

            {/* Document preview */}
            {previewUrl && isDocument && (
              <div className="cyber-card p-4">
                <p className="label-xs mb-3">Preview</p>
                <div className="rounded-xl overflow-hidden border border-[#1a2540] bg-[#060912]">
                  {file.type === 'application/pdf' ? (
                    <iframe src={previewUrl} title="Document preview" className="w-full" style={{ height: 420, border: 'none' }} />
                  ) : (
                    <img src={previewUrl} alt="document preview" className="w-full object-contain max-h-96" />
                  )}
                </div>
                <p className="text-slate-600 text-xs mt-2 font-mono">{file.name} · {(file.size / 1024).toFixed(1)} KB</p>
              </div>
            )}

            {/* Tips */}
            {tips?.length > 0 && (
              <div className={`rounded-xl border ${borderColor} ${bgColor} px-5 py-4`}>
                <p className={`label-xs ${accentColor} mb-3`}>What we detect</p>
                <ul className="space-y-1.5">
                  {tips.map((tip, i) => (
                    <li key={i} className="text-slate-400 text-sm flex items-start gap-2.5">
                      <span className={`mt-2 w-1 h-1 rounded-full shrink-0 ${accentColor} bg-current`} />
                      {tip}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Analyze button */}
            <button
              onClick={handleAnalyze}
              disabled={!canSubmit || busy}
              className="btn-primary w-full flex items-center justify-center gap-2.5 py-3.5"
            >
              {busy ? (
                <>
                  <Loader size={16} className="animate-spin" />
                  {step === 'uploading' ? `Uploading ${uploadProgress}%` : 'Analyzing with AI...'}
                </>
              ) : (
                <>
                  <Sparkles size={16} />
                  Run {title}
                </>
              )}
            </button>

            {/* Upload progress */}
            {uploading && (
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${uploadProgress}%` }} />
              </div>
            )}

            {/* Analyzing state */}
            {analyzing && (
              <div className="cyber-card p-4 flex items-center gap-3">
                <div className="spinner" style={{ width: 20, height: 20, borderWidth: 2 }} />
                <div>
                  <p className="text-white text-sm font-medium">Running AI analysis</p>
                  <p className="text-slate-600 text-xs">This may take a moment...</p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Results */}
        {report && (
          <div className="space-y-4 fade-in">
            {/* Result banner */}
            <div className={`rounded-xl border px-5 py-3.5 flex items-center justify-between ${
              report.prediction === 'fake'
                ? 'border-rose-500/25 bg-rose-500/5'
                : 'border-emerald-500/25 bg-emerald-500/5'
            }`}>
              <div className="flex items-center gap-3">
                <span className={`text-base font-bold tracking-wide ${report.prediction === 'fake' ? 'text-rose-400' : 'text-emerald-400'}`}>
                  {report.prediction?.toUpperCase()}
                </span>
                <span className="text-slate-500 text-sm">
                  {Math.round(report.confidence_score * 100)}% confidence · {report.risk_level} risk
                </span>
              </div>
              <button onClick={reset} className="flex items-center gap-1.5 text-slate-500 hover:text-white text-xs transition-colors">
                <RotateCcw size={12} /> New
              </button>
            </div>

            {/* Video in results */}
            {transcodedUrl && isVideo && (
              <div className="cyber-card p-4">
                <p className="label-xs mb-3">Analyzed Video</p>
                <video key={transcodedUrl} controls className="w-full rounded-xl" style={{ maxHeight: 280 }} src={transcodedUrl} />
              </div>
            )}

            {/* Audio in results */}
            {previewUrl && isAudio && (
              <div className="cyber-card p-4">
                <p className="label-xs mb-3">Analyzed Audio</p>
                <div className="rounded-xl border border-[#1a2540] bg-[#060912] px-4 py-3">
                  <audio key={previewUrl} controls className="w-full" style={{ accentColor: '#38bdf8' }}>
                    <source src={previewUrl} type={file?.type || 'audio/wav'} />
                  </audio>
                </div>
              </div>
            )}

            {/* Document in results */}
            {previewUrl && isDocument && (
              <div className="cyber-card p-4">
                <p className="label-xs mb-3">Analyzed Document</p>
                <div className="rounded-xl overflow-hidden border border-[#1a2540] bg-[#060912]">
                  {file?.type === 'application/pdf' ? (
                    <iframe src={previewUrl} title="Document" className="w-full" style={{ height: 360, border: 'none' }} />
                  ) : (
                    <img src={previewUrl} alt="document" className="w-full object-contain max-h-80" />
                  )}
                </div>
              </div>
            )}

            <AnalysisViewer report={report} />

            <button
              onClick={() => navigate(`/report/${report.report_id}`)}
              className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl border border-sky-400/25 text-sky-400 text-sm hover:bg-sky-400/5 transition-colors"
            >
              View Full Report <ArrowRight size={13} />
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
