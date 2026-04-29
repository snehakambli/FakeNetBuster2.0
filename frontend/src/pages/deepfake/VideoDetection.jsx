import React from 'react'
import { Video } from 'lucide-react'
import AnalysisPage from '../../components/AnalysisPage'

export default function VideoDetection() {
  return (
    <AnalysisPage
      title="Video Detection"
      subtitle="Detect deepfake videos using CNN frame analysis + LSTM temporal modeling"
      icon={Video}
      accentColor="text-purple-400"
      borderColor="border-purple-400/30"
      bgColor="bg-purple-400/10"
      acceptedTypes={{ 'video/*': ['.mp4', '.mov', '.avi'] }}
      acceptedExts="MP4, MOV, AVI"
      contentTypeHint="video"
      tips={[
        'Frame-by-frame CNN feature extraction',
        'Temporal inconsistency detection via LSTM',
        'Suspicious frame timestamps highlighted',
        'Face swap and face reenactment detection',
        'Attention-weighted frame suspicion scores',
      ]}
    />
  )
}
