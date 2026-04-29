import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Activity, AlertTriangle, CheckCircle, TrendingUp, RefreshCw, Image, Video, Mic, Newspaper, CreditCard } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import ReportCard from '../components/ReportCard'
import { getHistory, deleteReport } from '../services/api'
import toast from 'react-hot-toast'

const TABS = [
  { key: 'all',      label: 'All',       icon: Activity  },
  { key: 'image',    label: 'Images',    icon: Image     },
  { key: 'video',    label: 'Videos',    icon: Video     },
  { key: 'audio',    label: 'Audio',     icon: Mic       },
  { key: 'news',     label: 'News',      icon: Newspaper },
  { key: 'document', label: 'Documents', icon: CreditCard},
]

const LINKS = {
  image: '/deepfake/image', video: '/deepfake/video', audio: '/deepfake/audio',
  news: '/fake/news', document: '/fake/document',
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-[#0d1225] border border-[#1a2540] rounded-xl px-3 py-2 text-xs shadow-xl">
      <p className="text-slate-400 capitalize">{label}</p>
      <p className="text-sky-400 font-semibold mt-0.5">{payload[0].value} analyses</p>
    </div>
  )
}

export default function Dashboard() {
  const [reports, setReports]   = useState([])
  const [loading, setLoading]   = useState(true)
  const [activeTab, setActiveTab] = useState('all')

  const fetchReports = async () => {
    setLoading(true)
    try {
      const data = await getHistory(100)
      setReports(data.reports || [])
    } catch {
      toast.error('Failed to load reports')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchReports() }, [])

  const handleDelete = async (id) => {
    try {
      await deleteReport(id)
      setReports(prev => prev.filter(r => r.report_id !== id))
      toast.success('Report deleted')
    } catch {
      toast.error('Failed to delete report')
    }
  }

  const total         = reports.length
  const fakeCount     = reports.filter(r => r.prediction === 'fake').length
  const realCount     = total - fakeCount
  const criticalCount = reports.filter(r => r.risk_level === 'CRITICAL').length

  const typeData = ['image','video','audio','news','document'].map(type => ({
    name: type, count: reports.filter(r => r.content_type === type).length,
  })).filter(d => d.count > 0)

  const pieData = [
    { name: 'Fake', value: fakeCount, color: '#f87171' },
    { name: 'Real', value: realCount, color: '#34d399' },
  ].filter(d => d.value > 0)

  const filtered = activeTab === 'all' ? reports : reports.filter(r => r.content_type === activeTab)

  const stats = [
    { label: 'Total Analyses', value: total,         icon: Activity,      color: 'text-sky-400',     bg: 'bg-sky-400/8',     border: 'border-sky-400/20' },
    { label: 'Fake Detected',  value: fakeCount,     icon: AlertTriangle, color: 'text-rose-400',    bg: 'bg-rose-400/8',    border: 'border-rose-400/20' },
    { label: 'Real Content',   value: realCount,     icon: CheckCircle,   color: 'text-emerald-400', bg: 'bg-emerald-400/8', border: 'border-emerald-400/20' },
    { label: 'Critical Risk',  value: criticalCount, icon: TrendingUp,    color: 'text-orange-400',  bg: 'bg-orange-400/8',  border: 'border-orange-400/20' },
  ]

  return (
    <div className="min-h-screen cyber-grid pt-24 pb-16 px-4">
      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-2xl font-bold text-white">Dashboard</h1>
            <p className="text-slate-500 text-sm mt-0.5">Analysis history &amp; statistics</p>
          </div>
          <button
            onClick={fetchReports}
            className="flex items-center gap-2 px-4 py-2 rounded-xl border border-[#1a2540] text-slate-500 hover:text-white hover:border-[#243050] transition-all text-sm"
          >
            <RefreshCw size={13} className={loading ? 'animate-spin' : ''} /> Refresh
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-8">
          {stats.map(({ label, value, icon: Icon, color, bg, border }) => (
            <div key={label} className={`cyber-card p-5 border ${border}`}>
              <div className="flex items-center justify-between mb-3">
                <span className="text-slate-600 text-xs">{label}</span>
                <div className={`w-7 h-7 rounded-lg ${bg} flex items-center justify-center`}>
                  <Icon size={13} className={color} />
                </div>
              </div>
              <p className={`stat-number ${color}`}>{value}</p>
            </div>
          ))}
        </div>

        {/* Charts */}
        {total > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
            <div className="cyber-card p-5">
              <h3 className="label-xs mb-5">Analyses by Modality</h3>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={typeData} barSize={28}>
                  <XAxis dataKey="name" tick={{ fill: '#475569', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: '#475569', fontSize: 11 }} axisLine={false} tickLine={false} width={24} />
                  <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(56,189,248,0.04)' }} />
                  <Bar dataKey="count" fill="#38bdf8" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="cyber-card p-5">
              <h3 className="label-xs mb-5">Real vs Fake</h3>
              <ResponsiveContainer width="100%" height={160}>
                <PieChart>
                  <Pie data={pieData} cx="50%" cy="50%" innerRadius={44} outerRadius={70} dataKey="value" paddingAngle={4} strokeWidth={0}>
                    {pieData.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                  </Pie>
                  <Tooltip content={({ active, payload }) =>
                    active && payload?.length ? (
                      <div className="bg-[#0d1225] border border-[#1a2540] rounded-xl px-3 py-2 text-xs shadow-xl">
                        <p style={{ color: payload[0].payload.color }} className="font-semibold">{payload[0].name}: {payload[0].value}</p>
                      </div>
                    ) : null
                  } />
                </PieChart>
              </ResponsiveContainer>
              <div className="flex justify-center gap-5 mt-1">
                {pieData.map(({ name, color, value }) => (
                  <div key={name} className="flex items-center gap-1.5 text-xs text-slate-500">
                    <span className="w-2 h-2 rounded-full" style={{ background: color }} />
                    {name}: <span className="text-white font-medium">{value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Tabs */}
        <div className="flex items-center gap-1 flex-wrap mb-4">
          {TABS.map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              onClick={() => setActiveTab(key)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === key
                  ? 'bg-sky-400/10 text-sky-400 border border-sky-400/25'
                  : 'text-slate-500 hover:text-slate-300 border border-transparent hover:border-[#1a2540]'
              }`}
            >
              <Icon size={11} /> {label}
              {key !== 'all' && (
                <span className="text-slate-700 ml-0.5">
                  ({reports.filter(r => r.content_type === key).length})
                </span>
              )}
            </button>
          ))}
        </div>

        {/* List */}
        {loading ? (
          <div className="flex justify-center py-16"><div className="spinner" /></div>
        ) : filtered.length === 0 ? (
          <div className="cyber-card p-14 text-center">
            <Activity size={36} className="text-slate-700 mx-auto mb-3" />
            <p className="text-slate-500 text-sm mb-5">
              {activeTab === 'all' ? 'No analyses yet.' : `No ${activeTab} analyses yet.`}
            </p>
            {LINKS[activeTab] && (
              <Link to={LINKS[activeTab]} className="btn-primary inline-flex items-center gap-2 text-sm px-5 py-2">
                Run {activeTab} analysis →
              </Link>
            )}
          </div>
        ) : (
          <div className="space-y-2">
            {filtered.map(report => (
              <ReportCard key={report.report_id} report={report} onDelete={handleDelete} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
