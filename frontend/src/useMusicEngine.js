import { useCallback, useEffect, useRef, useState } from 'react'

const MOODS = ['joyful', 'tender', 'tense', 'sombre', 'unease', 'neutral']
const FADE_SECONDS = 2

// Loads all six mood tracks once, plays them all in a silent loop, and
// crossfades between them by ramping GainNodes instead of restarting playback.
export function useMusicEngine() {
  const contextRef = useRef(null)
  const gainsRef = useRef({})
  const activeMoodRef = useRef(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    const context = new (window.AudioContext || window.webkitAudioContext)()
    contextRef.current = context
    let cancelled = false
    const sources = []

    async function loadTrack(mood) {
      const response = await fetch(`/music/${mood}.mp3`)
      const arrayBuffer = await response.arrayBuffer()
      const buffer = await context.decodeAudioData(arrayBuffer)
      if (cancelled) return

      const gain = context.createGain()
      gain.gain.value = 0
      gain.connect(context.destination)

      const source = context.createBufferSource()
      source.buffer = buffer
      source.loop = true
      source.connect(gain)
      source.start(0)

      gainsRef.current[mood] = gain
      sources.push(source)
    }

    Promise.all(MOODS.map(loadTrack)).then(() => {
      if (!cancelled) setReady(true)
    })

    return () => {
      cancelled = true
      sources.forEach((source) => {
        try {
          source.stop()
        } catch {
          // already stopped
        }
      })
      gainsRef.current = {}
      contextRef.current = null
      context.close()
    }
  }, [])

  const start = useCallback(() => {
    const context = contextRef.current
    const neutralGain = gainsRef.current.neutral
    if (!context || !neutralGain) return

    context.resume()
    neutralGain.gain.cancelScheduledValues(context.currentTime)
    neutralGain.gain.setValueAtTime(1, context.currentTime)
    activeMoodRef.current = 'neutral'
  }, [])

  const crossfadeTo = useCallback((mood) => {
    const context = contextRef.current
    const targetGain = gainsRef.current[mood]
    if (!context || !targetGain) return
    if (activeMoodRef.current === mood) return

    const now = context.currentTime
    const previousGain = gainsRef.current[activeMoodRef.current]

    if (previousGain) {
      previousGain.gain.cancelScheduledValues(now)
      previousGain.gain.setValueAtTime(previousGain.gain.value, now)
      previousGain.gain.linearRampToValueAtTime(0, now + FADE_SECONDS)
    }

    targetGain.gain.cancelScheduledValues(now)
    targetGain.gain.setValueAtTime(targetGain.gain.value, now)
    targetGain.gain.linearRampToValueAtTime(1, now + FADE_SECONDS)

    activeMoodRef.current = mood
  }, [])

  return { ready, start, crossfadeTo }
}
