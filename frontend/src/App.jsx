import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import Dashboard from './pages/Dashboard'
import Report from './pages/Report'

// Deepfake detection pages
import ImageDetection from './pages/deepfake/ImageDetection'
import VideoDetection from './pages/deepfake/VideoDetection'
import AudioDetection from './pages/deepfake/AudioDetection'

// Fake content detection pages
import NewsDetection from './pages/fake/NewsDetection'
import DocumentDetection from './pages/fake/DocumentDetection'

export default function App() {
  return (
    <>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />

        {/* Deepfake detection */}
        <Route path="/deepfake/image" element={<ImageDetection />} />
        <Route path="/deepfake/video" element={<VideoDetection />} />
        <Route path="/deepfake/audio" element={<AudioDetection />} />

        {/* Fake content detection */}
        <Route path="/fake/news" element={<NewsDetection />} />
        <Route path="/fake/document" element={<DocumentDetection />} />

        {/* Dashboard & reports */}
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/reports" element={<Dashboard />} />
        <Route path="/report/:id" element={<Report />} />

        {/* Legacy redirect */}
        <Route path="/upload" element={<Navigate to="/" replace />} />
        <Route path="/report" element={<Navigate to="/reports" replace />} />
      </Routes>

      <Toaster
        position="bottom-right"
        toastOptions={{
          style: {
            background: '#0d1225',
            color: '#f1f5f9',
            border: '1px solid #1a2540',
            borderRadius: '12px',
            fontSize: '13px',
            fontFamily: 'Inter, system-ui, sans-serif',
            boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
          },
          success: { iconTheme: { primary: '#34d399', secondary: '#0d1225' } },
          error:   { iconTheme: { primary: '#f87171', secondary: '#0d1225' } },
        }}
      />
    </>
  )
}
