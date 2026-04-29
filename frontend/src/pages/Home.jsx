import React from 'react'
import { Link } from 'react-router-dom'
import { Zap, Image, Video, Mic, Newspaper, CreditCard, ArrowRight, Cpu, Eye, Lock } from 'lucide-react'
import useScrollReveal from '../hooks/useScrollReveal'

/* ── Single element reveal ── */
function Reveal({ children, className = '', delay = '', variant = 'reveal' }) {
  const [ref, visible] = useScrollReveal()
  return (
    <div ref={ref} className={`${variant} ${visible ? 'reveal-visible' : ''} ${delay} ${className}`}>
      {children}
    </div>
  )
}

/**
 * RevealGroup — one IntersectionObserver on the wrapper,
 * all direct children animate together with stagger delays.
 * This prevents inconsistent re-entry timing on repeat scrolls.
 */
function RevealGroup({ children, className = '', childClass = 'reveal', delays = [] }) {
  const [ref, visible] = useScrollReveal()
  return (
    <div ref={ref} className={className}>
      {React.Children.map(children, (child, i) =>
        React.cloneElement(child, {
          className: [
            child.props.className || '',
            childClass,
            visible ? 'reveal-visible' : '',
            delays[i] || '',
          ].join(' ').trim(),
        })
      )}
    </div>
  )
}

/* ── Card data ── */
const deepfakeCards = [
  {
    icon: Image, title: 'Image Detection', tag: 'CNN + Frequency',
    desc: 'GAN fingerprints, pixel inconsistencies, frequency artifacts with GradCAM heatmaps.',
    color: 'text-sky-400', bg: 'bg-sky-400/8', border: 'border-sky-400/20',
    glow: 'hover:shadow-[0_0_40px_rgba(56,189,248,0.13)]', to: '/deepfake/image',
  },
  {
    icon: Video, title: 'Video Detection', tag: 'CNN + LSTM',
    desc: 'Temporal inconsistency, face swap detection, suspicious frame timestamps.',
    color: 'text-violet-400', bg: 'bg-violet-400/8', border: 'border-violet-400/20',
    glow: 'hover:shadow-[0_0_40px_rgba(167,139,250,0.13)]', to: '/deepfake/video',
  },
  {
    icon: Mic, title: 'Audio Detection', tag: 'CNN + GRU',
    desc: 'Voice cloning artifacts, pitch anomalies, mel-spectrogram inconsistencies.',
    color: 'text-emerald-400', bg: 'bg-emerald-400/8', border: 'border-emerald-400/20',
    glow: 'hover:shadow-[0_0_40px_rgba(52,211,153,0.13)]', to: '/deepfake/audio',
  },
]

const fakeCards = [
  {
    icon: Newspaper, title: 'News Detection', tag: 'Transformer',
    desc: 'Clickbait patterns, semantic contradictions, multi-API fact-checking with LLM reasoning.',
    color: 'text-amber-400', bg: 'bg-amber-400/8', border: 'border-amber-400/20',
    glow: 'hover:shadow-[0_0_40px_rgba(251,191,36,0.13)]', to: '/fake/news',
  },
  {
    icon: CreditCard, title: 'Document Detection', tag: 'CNN + NLP',
    desc: 'ELA forgery analysis, font inconsistencies, fake seals, OCR text verification.',
    color: 'text-rose-400', bg: 'bg-rose-400/8', border: 'border-rose-400/20',
    glow: 'hover:shadow-[0_0_40px_rgba(248,113,113,0.13)]', to: '/fake/document',
  },
]

function DetectionCard({ icon: Icon, title, desc, color, bg, border, glow, to, tag, className = '' }) {
  return (
    <Link to={to} className={`cyber-card cyber-card-glow p-7 border ${border} ${glow} flex flex-col group transition-all duration-300 h-full ${className}`}>
      <div className="flex items-start justify-between gap-2 mb-6">
        <div className={`w-12 h-12 rounded-xl ${bg} border ${border} flex items-center justify-center shrink-0`}>
          <Icon size={22} className={color} />
        </div>
        <span className={`tag-pill ${border} ${color} opacity-80 mt-1`}>{tag}</span>
      </div>
      <h3 className="text-white font-bold text-lg mb-2.5">{title}</h3>
      <p className="text-slate-500 text-base leading-relaxed flex-1">{desc}</p>
      <div className={`mt-6 flex items-center gap-2 text-sm font-semibold ${color} group-hover:gap-3 transition-all duration-200`}>
        Analyze now <ArrowRight size={14} />
      </div>
    </Link>
  )
}

const steps = [
  { n: '01', title: 'Choose a modality', desc: 'Pick from Image, Video, Audio, News, or Document detection.' },
  { n: '02', title: 'Upload or paste',   desc: 'Drag & drop a file or paste text / URL on the detection page.' },
  { n: '03', title: 'AI runs analysis',  desc: 'The dedicated model runs inference with full explainability.' },
  { n: '04', title: 'View the report',   desc: 'Confidence score, anomalies, visualizations, and generation method.' },
]

const pillars = [
  { icon: Cpu,  title: 'Local inference', desc: 'No cloud APIs. Runs entirely on your GPU.' },
  { icon: Eye,  title: 'Explainable AI',  desc: 'GradCAM, ELA, attention maps — every verdict has evidence.' },
  { icon: Lock, title: 'Privacy-first',   desc: 'Files never leave your machine.' },
]

const stagger = ['reveal-delay-1', 'reveal-delay-2', 'reveal-delay-3', 'reveal-delay-4']

export default function Home() {
  return (
    <div className="min-h-screen cyber-grid relative">
      <div className="scan-line" />

      {/* ── Hero ── */}
      <section className="relative pt-32 pb-10 px-6 max-w-7xl mx-auto text-center">

        <Reveal className="flex justify-center">
          <div className="inline-flex items-center gap-2 px-5 py-2 rounded-full border border-sky-400/25 bg-sky-400/5 text-sky-400 text-sm font-semibold tracking-wide">
            <Zap size={13} /> AI-Powered · Multi-Modal · Fully Local
          </div>
        </Reveal>

        <Reveal className="mt-7" delay="reveal-delay-1">
          <h1 className="text-6xl sm:text-8xl font-extrabold leading-[1.0] tracking-tight">
            <span className="text-white">FakeNet</span>
            <span className="gradient-text">Buster</span>
            <span className="text-slate-600 font-light text-5xl sm:text-6xl align-baseline ml-2">2.0</span>
          </h1>
        </Reveal>

        <Reveal className="mt-6" delay="reveal-delay-2">
          <p className="text-slate-400 text-lg sm:text-xl max-w-2xl mx-auto leading-relaxed">
            Five specialized AI models. One platform. Detect deepfakes and misinformation across every modality — with full explainability.
          </p>
        </Reveal>

        {/* Stats — all 4 observed together, stagger via CSS delay */}
        <RevealGroup
          className="grid grid-cols-2 sm:grid-cols-4 gap-4 max-w-2xl mx-auto mt-10 mb-16"
          childClass="reveal-scale"
          delays={stagger}
        >
          {[
            { label: 'Modalities',     value: '5' },
            { label: 'AI Models',      value: '5' },
            { label: 'Explainability', value: 'GradCAM' },
            { label: 'Processing',     value: 'GPU Local' },
          ].map(({ label, value }) => (
            <div key={label} className="cyber-card p-5 text-center">
              <p className="gradient-text font-bold text-xl">{value}</p>
              <p className="text-slate-600 text-sm mt-1">{label}</p>
            </div>
          ))}
        </RevealGroup>
      </section>

      {/* ── Deepfake Detection ── */}
      <section className="py-8 px-6 max-w-7xl mx-auto">
        <Reveal>
          <div className="flex items-center gap-4 mb-8">
            <div className="section-bar h-9 bg-gradient-to-b from-sky-400 to-violet-500" />
            <div>
              <h2 className="text-2xl font-bold text-white">Deepfake Detection</h2>
              <p className="text-slate-500 text-sm mt-0.5">Images · Videos · Audio</p>
            </div>
          </div>
        </Reveal>

        {/* All 3 cards observed as one group */}
        <RevealGroup
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5"
          childClass="reveal"
          delays={stagger}
        >
          {deepfakeCards.map((card) => (
            <DetectionCard key={card.to} {...card} />
          ))}
        </RevealGroup>
      </section>

      {/* ── Fake Content Detection ── */}
      <section className="py-8 px-6 max-w-7xl mx-auto">
        <Reveal>
          <div className="flex items-center gap-4 mb-8">
            <div className="section-bar h-9 bg-gradient-to-b from-amber-400 to-rose-500" />
            <div>
              <h2 className="text-2xl font-bold text-white">Fake Content Detection</h2>
              <p className="text-slate-500 text-sm mt-0.5">News · Documents</p>
            </div>
          </div>
        </Reveal>

        <RevealGroup
          className="grid grid-cols-1 sm:grid-cols-2 gap-5"
          childClass="reveal"
          delays={stagger}
        >
          {fakeCards.map((card) => (
            <DetectionCard key={card.to} {...card} />
          ))}
        </RevealGroup>
      </section>

      {/* ── Pillars ── */}
      <section className="py-10 px-6 max-w-7xl mx-auto">
        <RevealGroup
          className="grid grid-cols-1 sm:grid-cols-3 gap-5"
          childClass="reveal"
          delays={stagger}
        >
          {pillars.map(({ icon: Icon, title, desc }) => (
            <div key={title} className="cyber-card p-6 flex items-start gap-5 h-full">
              <div className="w-11 h-11 rounded-xl bg-sky-400/8 border border-sky-400/20 flex items-center justify-center shrink-0">
                <Icon size={18} className="text-sky-400" />
              </div>
              <div>
                <p className="text-white font-bold text-base">{title}</p>
                <p className="text-slate-500 text-sm mt-1.5 leading-relaxed">{desc}</p>
              </div>
            </div>
          ))}
        </RevealGroup>
      </section>

      {/* ── How it works ── */}
      <section className="py-8 px-6 max-w-4xl mx-auto">
        <Reveal>
          <div className="flex items-center gap-4 mb-8">
            <div className="section-bar h-9 bg-sky-400" />
            <h2 className="text-2xl font-bold text-white">How it works</h2>
          </div>
        </Reveal>

        <RevealGroup
          className="grid grid-cols-1 sm:grid-cols-2 gap-4"
          childClass="reveal"
          delays={stagger}
        >
          {steps.map(({ n, title, desc }) => (
            <div key={n} className="cyber-card p-6 flex items-start gap-5 h-full">
              <span className="font-mono text-3xl font-bold text-sky-400/20 shrink-0 w-10 leading-none">{n}</span>
              <div>
                <p className="text-white font-bold text-base">{title}</p>
                <p className="text-slate-500 text-sm mt-1.5 leading-relaxed">{desc}</p>
              </div>
            </div>
          ))}
        </RevealGroup>
      </section>

      <div className="h-24" />
    </div>
  )
}
