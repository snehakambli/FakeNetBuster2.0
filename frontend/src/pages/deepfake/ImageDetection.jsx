import React from 'react'
import { Image } from 'lucide-react'
import AnalysisPage from '../../components/AnalysisPage'

export default function ImageDetection() {
  return (
    <AnalysisPage
      title="Image Detection"
      subtitle="Detect AI-generated or manipulated images using dual-branch CNN analysis"
      icon={Image}
      accentColor="text-blue-400"
      borderColor="border-blue-400/30"
      bgColor="bg-blue-400/10"
      acceptedTypes={{ 'image/*': ['.jpg', '.jpeg', '.png'] }}
      acceptedExts="JPG, JPEG, PNG"
      contentTypeHint="image"
      tips={[
        'GAN fingerprints and frequency domain artifacts',
        'Pixel-level inconsistencies and noise patterns',
        'Color blending mismatches at face boundaries',
        'GradCAM heatmap highlighting suspicious regions',
        'Possible generation method (GAN / Diffusion)',
      ]}
    />
  )
}
