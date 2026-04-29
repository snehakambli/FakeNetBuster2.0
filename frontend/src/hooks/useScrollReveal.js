import { useEffect, useRef, useState } from 'react'

/**
 * Returns a ref and a boolean `visible`.
 * Toggles every time the element enters/leaves the viewport —
 * so the animation replays each time you scroll past.
 */
export default function useScrollReveal(options = {}) {
  const ref = useRef(null)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return

    const observer = new IntersectionObserver(
      ([entry]) => setVisible(entry.isIntersecting),
      { threshold: 0.1, rootMargin: '0px 0px -40px 0px', ...options }
    )

    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  return [ref, visible]
}
