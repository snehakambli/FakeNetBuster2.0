import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 120000, // 2 min for large files
})

// Request interceptor
api.interceptors.request.use(config => {
  return config
}, error => Promise.reject(error))

// Response interceptor
api.interceptors.response.use(
  response => response.data,
  error => {
    const message = error.response?.data?.detail || error.message || 'Request failed'
    return Promise.reject(new Error(message))
  }
)

export const uploadFile = async (file, onProgress) => {
  const formData = new FormData()
  formData.append('file', file)
  return api.post('/upload/', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: e => {
      if (onProgress) onProgress(Math.round((e.loaded * 100) / e.total))
    }
  })
}

export const analyzeFile = async (filePath, contentTypeHint = null) => {
  return api.post('/analyze/file', {
    file_path: filePath,
    content_type_hint: contentTypeHint
  })
}

export const analyzeNews = async (text = null, url = null) => {
  return api.post('/analyze/news', { text, url })
}

export const getReport = async (reportId) => {
  return api.get(`/report/${reportId}`)
}

export const getHistory = async (limit = 50) => {
  return api.get(`/report/history?limit=${limit}`)
}

export const deleteReport = async (reportId) => {
  return api.delete(`/report/${reportId}`)
}

export const getPreviewUrl = (filePath) => {
  const base = import.meta.env.VITE_API_URL || 'http://localhost:8000'
  return `${base}/preview/video?path=${encodeURIComponent(filePath)}`
}

export default api
