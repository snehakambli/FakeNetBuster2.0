import React from 'react'
import { CreditCard } from 'lucide-react'
import AnalysisPage from '../../components/AnalysisPage'

export default function DocumentDetection() {
  return (
    <AnalysisPage
      title="Document Detection"
      subtitle="Detect forged IDs, certificates, and documents using CNN visual analysis + OCR"
      icon={CreditCard}
      accentColor="text-red-400"
      borderColor="border-red-400/30"
      bgColor="bg-red-400/10"
      acceptedTypes={{
        'image/*': ['.jpg', '.jpeg', '.png'],
        'application/pdf': ['.pdf'],
      }}
      acceptedExts="JPG, PNG, PDF"
      contentTypeHint="document"
      tips={[
        'Error Level Analysis (ELA) for copy-paste forgery',
        'Font inconsistency detection across regions',
        'Fake seal and stamp identification',
        'OCR text authenticity verification',
        'Layout anomaly detection',
      ]}
    />
  )
}
