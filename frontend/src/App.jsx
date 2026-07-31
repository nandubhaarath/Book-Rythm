import { useEffect, useRef, useState } from 'react'
import './App.css'
import { useMusicEngine } from './useMusicEngine'

const UPLOAD_URL = 'http://127.0.0.1:8000/upload'

const MOOD_COLORS = {
  joyful: '#c98a2b',
  neutral: '#6b6b6b',
  sombre: '#3d4a5c',
  tender: '#b5677d',
  tense: '#9c3b3b',
  unease: '#5b4b8a',
}

function MoodBadge({ mood, confidence }) {
  if (!mood) return null
  const color = MOOD_COLORS[mood] ?? '#6b6b6b'
  return (
    <div className="mood-badge">
      <span className="mood-dot" style={{ background: color }} />
      <span className="mood-name">{mood}</span>
      {typeof confidence === 'number' && (
        <span className="mood-confidence">{Math.round(confidence * 100)}%</span>
      )}
    </div>
  )
}

function UploadScreen({ onFileSelected, error }) {
  return (
    <div className="screen upload-screen">
      <h1>Book Rythm</h1>
      <p className="tagline">Upload a PDF and read it to a mood-matched soundtrack.</p>
      <label className="file-button">
        Choose a PDF
        <input
          type="file"
          accept="application/pdf,.pdf"
          onChange={onFileSelected}
          hidden
        />
      </label>
      {error && <p className="error">{error}</p>}
    </div>
  )
}

function LoadingScreen() {
  return (
    <div className="screen loading-screen">
      <div className="spinner" />
      <p>Reading your book and scoring its mood…</p>
      <p className="subtext">This can take a little while for a whole book.</p>
    </div>
  )
}

export default function App() {
  const [status, setStatus] = useState('upload') // 'upload' | 'loading' | 'reading'
  const [paragraphs, setParagraphs] = useState([])
  const [activeIndex, setActiveIndex] = useState(0)
  const [error, setError] = useState(null)
  const [started, setStarted] = useState(false)

  const paragraphRefs = useRef([])
  const { ready: musicReady, start, crossfadeTo } = useMusicEngine()

  const handleFileSelected = async (event) => {
    const file = event.target.files?.[0]
    if (!file) return

    setError(null)
    setStatus('loading')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(UPLOAD_URL, {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) {
        throw new Error(`Upload failed (${response.status})`)
      }
      const data = await response.json()
      paragraphRefs.current = []
      setParagraphs(data.paragraphs)
      setActiveIndex(0)
      setStatus('reading')
    } catch (err) {
      setError(err.message || 'Something went wrong uploading the file.')
      setStatus('upload')
    }
  }

  useEffect(() => {
    if (status !== 'reading') return undefined

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveIndex(Number(entry.target.dataset.index))
          }
        })
      },
      { rootMargin: '-50% 0px -50% 0px', threshold: 0 }
    )

    paragraphRefs.current.forEach((el) => {
      if (el) observer.observe(el)
    })

    return () => observer.disconnect()
  }, [status, paragraphs])

  const activeMood = paragraphs[activeIndex]?.mood

  useEffect(() => {
    if (!started || !activeMood) return
    crossfadeTo(activeMood)
  }, [started, activeMood, crossfadeTo])

  if (status === 'upload') {
    return <UploadScreen onFileSelected={handleFileSelected} error={error} />
  }

  if (status === 'loading') {
    return <LoadingScreen />
  }

  const active = paragraphs[activeIndex]

  return (
    <div className="screen reading-screen">
      {!started && (
        <button
          type="button"
          className="start-button"
          disabled={!musicReady}
          onClick={() => {
            start()
            setStarted(true)
          }}
        >
          {musicReady ? 'Start reading' : 'Loading music…'}
        </button>
      )}
      <MoodBadge mood={active?.mood} confidence={active?.confidence} />
      <div className="reading-column">
        {paragraphs.map((paragraph, index) => (
          <p
            key={index}
            data-index={index}
            ref={(el) => {
              paragraphRefs.current[index] = el
            }}
            className="paragraph"
          >
            {paragraph.text}
          </p>
        ))}
      </div>
    </div>
  )
}
