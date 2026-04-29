/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        cyber: {
          bg:      '#060912',
          card:    '#0d1225',
          border:  '#1a2540',
          accent:  '#38bdf8',
          green:   '#34d399',
          red:     '#f87171',
          yellow:  '#fbbf24',
          purple:  '#a78bfa',
          orange:  '#fb923c',
        }
      },
      fontFamily: {
        sans: ['Josefin Sans', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
      },
      animation: {
        'pulse-slow':  'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin-slow':   'spin 2s linear infinite',
        'fade-in':     'fadeIn 0.35s ease-out',
        'slide-up':    'slideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
      },
      boxShadow: {
        'glow-sm':  '0 0 12px rgba(56, 189, 248, 0.2)',
        'glow':     '0 0 24px rgba(56, 189, 248, 0.25)',
        'glow-lg':  '0 0 48px rgba(56, 189, 248, 0.2)',
        'card':     '0 4px 24px rgba(0, 0, 0, 0.4)',
      },
      borderRadius: {
        'xl2': '14px',
      }
    }
  },
  plugins: []
}
