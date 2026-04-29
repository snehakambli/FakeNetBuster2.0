import React from 'react'
import { Mic } from 'lucide-react'
import AnalysisPage from '../../components/AnalysisPage'

export default function AudioDetection() {
  return (
    <AnalysisPage
      title="Audio Detection"
      subtitle="Detect voice cloning and synthetic speech using CNN + GRU spectrogram analysis"
      icon={Mic}
      accentColor="text-green-400"
      borderColor="border-green-400/30"
      bgColor="bg-green-400/10"
      acceptedTypes={{ 'audio/*': ['.wav', '.mp3'] }}
      acceptedExts="WAV, MP3"
      contentTypeHint="audio"
      tips={[
        'Mel-spectrogram anomaly detection',
        'Voice cloning artifact signatures',
        'Pitch anomaly and unnatural transitions',
        'Spectral inconsistency in high-frequency bands',
        'Temporal attention highlighting suspicious segments',
      ]}
    />
  )
}
