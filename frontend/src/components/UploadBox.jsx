import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, FileImage, FileVideo, FileAudio, FileText, X, CheckCircle } from 'lucide-react'

const DEFAULT_ACCEPT = {
  'image/*':        ['.jpg', '.jpeg', '.png'],
  'video/*':        ['.mp4', '.mov', '.avi'],
  'audio/*':        ['.wav', '.mp3'],
  'application/pdf':['.pdf'],
}

const getFileIcon = (type) => {
  if (!type) return FileText
  if (type.startsWith('image')) return FileImage
  if (type.startsWith('video')) return FileVideo
  if (type.startsWith('audio')) return FileAudio
  return FileText
}

const formatSize = (bytes) =>
  bytes < 1024 * 1024
    ? `${(bytes / 1024).toFixed(1)} KB`
    : `${(bytes / (1024 * 1024)).toFixed(1)} MB`

export default function UploadBox({ onFileSelect, uploading, uploadProgress, acceptedTypes, acceptedExts }) {
  const [selectedFile, setSelectedFile] = useState(null)
  const accept       = acceptedTypes || DEFAULT_ACCEPT
  const extsDisplay  = acceptedExts  || 'JPG, PNG, MP4, WAV, MP3, PDF'

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      const f = acceptedFiles[0]
      setSelectedFile(f)
      onFileSelect(f)
    }
  }, [onFileSelect])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept, maxFiles: 1, disabled: uploading,
  })

  const clearFile = (e) => {
    e.stopPropagation()
    setSelectedFile(null)
    onFileSelect(null)
  }

  const FileIcon = selectedFile ? getFileIcon(selectedFile.type) : Upload

  return (
    <div className="w-full">
      <div
        {...getRootProps()}
        className={`relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-300 ${
          isDragActive
            ? 'border-sky-400 bg-sky-400/5 shadow-[0_0_32px_rgba(56,189,248,0.1)]'
            : selectedFile
            ? 'border-emerald-400/50 bg-emerald-400/5'
            : 'border-[#1a2540] hover:border-sky-400/35 hover:bg-sky-400/[0.02]'
        } ${uploading ? 'pointer-events-none opacity-50' : ''}`}
      >
        <input {...getInputProps()} />

        {selectedFile ? (
          <div className="flex flex-col items-center gap-3">
            <div className="relative">
              <div className="w-14 h-14 rounded-xl bg-emerald-400/10 border border-emerald-400/25 flex items-center justify-center">
                <FileIcon size={22} className="text-emerald-400" />
              </div>
              <div className="absolute -bottom-1 -right-1 w-5 h-5 rounded-full bg-emerald-400 flex items-center justify-center">
                <CheckCircle size={12} className="text-[#060912]" />
              </div>
            </div>
            <div>
              <p className="text-white font-medium text-sm">{selectedFile.name}</p>
              <p className="text-slate-500 text-xs mt-0.5">{formatSize(selectedFile.size)}</p>
            </div>
            {!uploading && (
              <button
                onClick={clearFile}
                className="absolute top-3 right-3 w-7 h-7 flex items-center justify-center rounded-lg text-slate-600 hover:text-rose-400 hover:bg-rose-400/10 transition-colors"
              >
                <X size={14} />
              </button>
            )}
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4">
            <div className={`w-14 h-14 rounded-xl border-2 flex items-center justify-center transition-all duration-300 ${
              isDragActive
                ? 'border-sky-400 bg-sky-400/10 shadow-[0_0_20px_rgba(56,189,248,0.2)]'
                : 'border-[#1a2540] bg-[#0b0f1e]'
            }`}>
              <Upload size={22} className={isDragActive ? 'text-sky-400' : 'text-slate-600'} />
            </div>
            <div>
              <p className="text-slate-200 font-medium text-sm">
                {isDragActive ? 'Drop it here' : 'Drag & drop or click to upload'}
              </p>
              <p className="text-slate-600 text-xs mt-1">{extsDisplay}</p>
            </div>
          </div>
        )}

        {uploading && (
          <div className="mt-5">
            <div className="flex justify-between text-xs text-slate-500 mb-1.5">
              <span>Uploading</span>
              <span className="font-mono">{uploadProgress}%</span>
            </div>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${uploadProgress}%` }} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
