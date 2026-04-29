import React, { useState, useEffect, useRef } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Shield, Activity, Menu, X, ChevronDown, Image, Video, Mic, Newspaper, CreditCard } from 'lucide-react'

const deepfakeLinks = [
  { to: '/deepfake/image', label: 'Image Detection',    icon: Image,      color: 'text-sky-400' },
  { to: '/deepfake/video', label: 'Video Detection',    icon: Video,      color: 'text-violet-400' },
  { to: '/deepfake/audio', label: 'Audio Detection',    icon: Mic,        color: 'text-emerald-400' },
]

const fakeLinks = [
  { to: '/fake/news',      label: 'News Detection',     icon: Newspaper,  color: 'text-amber-400' },
  { to: '/fake/document',  label: 'Document Detection', icon: CreditCard, color: 'text-rose-400' },
]

function DropdownMenu({ label, links, isOpen, onToggle, location }) {
  const ref = useRef(null)
  const isActive = links.some(l => location.pathname === l.to)

  useEffect(() => {
    const handler = (e) => { if (ref.current && !ref.current.contains(e.target)) onToggle(false) }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [onToggle])

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => onToggle(!isOpen)}
        className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold tracking-wide transition-all duration-200 ${
          isActive
            ? 'bg-sky-400/10 text-sky-400 border border-sky-400/25'
            : 'text-slate-400 hover:text-white hover:bg-white/[0.05]'
        }`}
      >
        {label}
        <ChevronDown size={13} className={`transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="absolute top-full left-0 mt-2 w-56 bg-[#0d1225] border border-[#1a2540] rounded-xl shadow-2xl shadow-black/70 overflow-hidden z-50 slide-up">
          {links.map(({ to, label: lbl, icon: Icon, color }) => (
            <Link
              key={to}
              to={to}
              onClick={() => onToggle(false)}
              className={`flex items-center gap-3 px-4 py-3 text-sm font-medium transition-colors hover:bg-white/[0.05] ${
                location.pathname === to ? 'text-sky-400 bg-sky-400/5' : 'text-slate-300'
              }`}
            >
              <Icon size={15} className={color} />
              {lbl}
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}

export default function Navbar() {
  const location = useLocation()
  const [scrolled, setScrolled]         = useState(false)
  const [menuOpen, setMenuOpen]         = useState(false)
  const [deepfakeOpen, setDeepfakeOpen] = useState(false)
  const [fakeOpen, setFakeOpen]         = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 16)
    window.addEventListener('scroll', onScroll)
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => {
    setDeepfakeOpen(false)
    setFakeOpen(false)
    setMenuOpen(false)
  }, [location.pathname])

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
      scrolled
        ? 'bg-[#060912]/95 backdrop-blur-xl border-b border-[#1a2540]'
        : 'bg-[#060912]/60 backdrop-blur-md'
    }`}>
      <div className="max-w-7xl mx-auto px-6 lg:px-10">
        <div className="flex items-center h-[68px] gap-8">

          {/* Logo — left anchored */}
          <Link to="/" className="flex items-center gap-2.5 shrink-0 group mr-auto md:mr-0">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-sky-400 to-blue-600 flex items-center justify-center shadow-[0_0_16px_rgba(56,189,248,0.3)] group-hover:shadow-[0_0_24px_rgba(56,189,248,0.45)] transition-shadow duration-300">
              <Shield size={17} className="text-white" />
            </div>
            <span className="font-bold text-[17px] tracking-tight leading-none">
              <span className="text-white">FakeNet</span>
              <span className="gradient-text">Buster</span>
              <span className="text-slate-600 font-light text-sm ml-1">2.0</span>
            </span>
          </Link>

          {/* Desktop nav — pushed to the right */}
          <div className="hidden md:flex items-center gap-1 ml-auto">
            <DropdownMenu
              label="Deepfake"
              links={deepfakeLinks}
              isOpen={deepfakeOpen}
              onToggle={setDeepfakeOpen}
              location={location}
            />
            <DropdownMenu
              label="Fake Content"
              links={fakeLinks}
              isOpen={fakeOpen}
              onToggle={setFakeOpen}
              location={location}
            />
            <Link
              to="/dashboard"
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold tracking-wide transition-all duration-200 ${
                location.pathname === '/dashboard'
                  ? 'bg-sky-400/10 text-sky-400 border border-sky-400/25'
                  : 'text-slate-400 hover:text-white hover:bg-white/[0.05]'
              }`}
            >
              <Activity size={15} />
              Dashboard
            </Link>
          </div>

          {/* Status dot — far right */}
          <div className="hidden md:flex items-center gap-2 text-xs text-slate-600 ml-6 shrink-0">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 pulse-dot" />
            Online
          </div>

          {/* Mobile toggle */}
          <button
            className="md:hidden ml-auto w-9 h-9 flex items-center justify-center rounded-lg text-slate-400 hover:text-white hover:bg-white/[0.05] transition-colors"
            onClick={() => setMenuOpen(!menuOpen)}
          >
            {menuOpen ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {menuOpen && (
        <div className="md:hidden bg-[#0d1225]/98 backdrop-blur-xl border-b border-[#1a2540] px-6 py-4 space-y-1 slide-up">
          <p className="label-xs px-2 py-1.5">Deepfake Detection</p>
          {deepfakeLinks.map(({ to, label, icon: Icon, color }) => (
            <Link key={to} to={to} className={`flex items-center gap-3 px-3 py-3 rounded-xl text-sm font-medium transition-colors ${
              location.pathname === to ? 'bg-sky-400/10 text-sky-400' : 'text-slate-400 hover:text-white hover:bg-white/[0.04]'
            }`}>
              <Icon size={15} className={color} />{label}
            </Link>
          ))}
          <p className="label-xs px-2 py-1.5 mt-2">Fake Content</p>
          {fakeLinks.map(({ to, label, icon: Icon, color }) => (
            <Link key={to} to={to} className={`flex items-center gap-3 px-3 py-3 rounded-xl text-sm font-medium transition-colors ${
              location.pathname === to ? 'bg-sky-400/10 text-sky-400' : 'text-slate-400 hover:text-white hover:bg-white/[0.04]'
            }`}>
              <Icon size={15} className={color} />{label}
            </Link>
          ))}
          <div className="border-t border-[#1a2540] mt-3 pt-3">
            <Link to="/dashboard" className={`flex items-center gap-3 px-3 py-3 rounded-xl text-sm font-medium transition-colors ${
              location.pathname === '/dashboard' ? 'bg-sky-400/10 text-sky-400' : 'text-slate-400 hover:text-white hover:bg-white/[0.04]'
            }`}>
              <Activity size={15} />Dashboard
            </Link>
          </div>
        </div>
      )}
    </nav>
  )
}
