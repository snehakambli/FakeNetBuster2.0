import React from 'react'
import { Newspaper } from 'lucide-react'
import AnalysisPage from '../../components/AnalysisPage'

export default function NewsDetection() {
  return (
    <AnalysisPage
      title="News Detection"
      subtitle="Detect fake news and misinformation using a custom Transformer model"
      icon={Newspaper}
      accentColor="text-yellow-400"
      borderColor="border-yellow-400/30"
      bgColor="bg-yellow-400/10"
      isTextInput
      textPlaceholder="Paste a news article, headline, or enter a URL (https://...) to analyze for misinformation..."
      contentTypeHint="news"
      tips={[
        'Clickbait language and sensationalist patterns',
        'Semantic contradictions and misleading claims',
        'Credibility indicator analysis',
        'Suspicious token highlighting via attention',
        'Multilingual support',
      ]}
    />
  )
}
