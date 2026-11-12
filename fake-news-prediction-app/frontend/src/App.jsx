import { useEffect, useState } from 'react'
import {
  Activity,
  ArrowRight,
  CheckCircle2,
  FileText,
  Info,
  LoaderCircle,
  Newspaper,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  XCircle,
} from 'lucide-react'
import { checkHealth, getModelMetrics, predictNews } from './api'

const percent = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`

function App() {
  const [text, setText] = useState('')
  const [title, setTitle] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [apiState, setApiState] = useState('checking')
  const [metrics, setMetrics] = useState(null)

  async function refreshStatus() {
    try {
      const health = await checkHealth()
      if (!health.model_ready) {
        setApiState('model-missing')
        setMetrics(null)
        return
      }
      setApiState('ready')
      try {
        const report = await getModelMetrics()
        setMetrics(report.metrics || null)
      } catch {
        setMetrics(null)
      }
    } catch {
      setApiState('offline')
      setMetrics(null)
    }
  }

  useEffect(() => {
    refreshStatus()
  }, [])

  async function handleSubmit(event) {
    event.preventDefault()
    setError('')
    setResult(null)
    setLoading(true)
    try {
      const prediction = await predictNews({ title: title.trim(), text: text.trim() })
      setResult(prediction)
      setApiState('ready')
    } catch (err) {
      setError(err.message || 'Something went wrong. Please try again.')
      if (err.message?.toLowerCase().includes('model')) setApiState('model-missing')
    } finally {
      setLoading(false)
    }
  }

  function resetForm() {
    setText('')
    setTitle('')
    setResult(null)
    setError('')
  }

  const isFake = result?.prediction === 1
  const confidence = result ? Math.round(result.confidence * 100) : 0
  const statusLabel = {
    checking: 'CHECKING API',
    ready: 'MODEL ONLINE',
    'model-missing': 'MODEL SETUP REQUIRED',
    offline: 'API OFFLINE',
  }[apiState]

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Verity home">
          <span className="brand-mark"><ShieldCheck size={20} /></span>
          <span>verity<span className="brand-dot">.</span></span>
        </a>
        <div className="topbar-right">
          <span className="topbar-label">AI-POWERED NEWS CHECK</span>
          <span className="topbar-icon"><Activity size={16} /></span>
        </div>
      </header>

      <main id="top">
        <section className="hero">
          <div className="eyebrow"><Sparkles size={14} /> THINK BEFORE YOU SHARE</div>
          <h1>Separate the facts<br /><span>from the noise.</span></h1>
          <p className="hero-copy">A machine-learning assistant that analyses a news headline and article text to help you think critically about what you read.</p>
          <div className="hero-meta">
            <span><span className={`pulse-dot ${apiState !== 'ready' ? 'pulse-dot-muted' : ''}`} /> SERVICE STATUS</span>
            <strong className={apiState === 'ready' ? 'status-ready' : 'status-offline'}>{statusLabel}</strong>
            {apiState !== 'ready' && <button className="refresh-status" type="button" onClick={refreshStatus}>Refresh</button>}
          </div>
        </section>

        <section className="workspace" aria-label="News analysis">
          <div className="form-panel">
            <div className="panel-heading">
              <div><span className="step-label">01 / INPUT</span><h2>Examine a story</h2></div>
              <div className="heading-icon"><Newspaper size={20} /></div>
            </div>
            <p className="panel-intro">Enter a headline and, when available, the article body. The model analyses both fields.</p>

            <form onSubmit={handleSubmit}>
              <div className="label-row"><label htmlFor="title">NEWS HEADLINE</label><span className="required">REQUIRED</span></div>
              <textarea id="title" value={title} onChange={(event) => setTitle(event.target.value)} placeholder="Paste the news headline..." rows={3} maxLength={500} required />
              <div className="input-footer"><span><FileText size={13} /> Headline</span><span>{title.length}/500</span></div>

              <div className="label-row"><label htmlFor="article-text">ARTICLE BODY</label><span className="optional">OPTIONAL · RECOMMENDED</span></div>
              <textarea id="article-text" value={text} onChange={(event) => setText(event.target.value)} placeholder="Paste the article body for a more informative prediction..." rows={7} maxLength={100000} />
              <div className="input-footer"><span><FileText size={13} /> Headline + article text</span><span>{text.length.toLocaleString()}/100,000</span></div>

              <button className="analyze-button" type="submit" disabled={loading || !title.trim()}>
                {loading ? <><LoaderCircle className="spin" size={17} /> Analysing story</> : <>Analyse article <ArrowRight size={17} /></>}
              </button>
            </form>
            {error && <div className="error-box" role="alert"><XCircle size={17} /><span>{error}</span></div>}
            {apiState === 'model-missing' && !error && <div className="setup-note"><Info size={17} /><span>The API is running, but the trained model is missing. Build the dataset and train the model using the README instructions.</span></div>}
            {apiState === 'offline' && !error && <div className="setup-note"><Info size={17} /><span>Cannot reach FastAPI. Start the backend at the configured API URL, then refresh the status.</span></div>}
          </div>

          <div className="result-panel">
            <div className="panel-heading">
              <div><span className="step-label">02 / RESULT</span><h2>Analysis report</h2></div>
              <div className="heading-icon"><Activity size={20} /></div>
            </div>
            {!result ? (
              <div className="empty-state">
                <div className="empty-graphic"><div className="graphic-ring ring-one" /><div className="graphic-ring ring-two" /><div className="graphic-center"><ShieldCheck size={34} /></div></div>
                <h3>Your report starts here</h3>
                <p>Submit a headline to see the model's prediction and class probabilities.</p>
                <div className="empty-divider"><span /> WAITING FOR INPUT <span /></div>
              </div>
            ) : (
              <div className="result-content" aria-live="polite">
                <div className={`verdict-icon ${isFake ? 'fake' : 'real'}`}>{isFake ? <XCircle size={27} /> : <CheckCircle2 size={27} />}</div>
                <span className="result-kicker">MODEL PREDICTION</span>
                <h3 className={`verdict ${isFake ? 'fake-text' : 'real-text'}`}>{result.label}</h3>
                <p className="verdict-copy">{isFake ? 'The model classified this article as likely fake.' : 'The model classified this article as likely real.'}</p>
                <div className="confidence-block">
                  <div className="confidence-head"><span>MODEL CONFIDENCE</span><strong>{confidence}%</strong></div>
                  <div className="confidence-track"><div className={`confidence-fill ${isFake ? 'fake-fill' : 'real-fill'}`} style={{ width: `${confidence}%` }} /></div>
                </div>
                <div className="probability-grid">
                  <div><span>REAL NEWS</span><strong>{percent(result.probability_real)}</strong></div>
                  <div><span>FAKE NEWS</span><strong>{percent(result.probability_fake)}</strong></div>
                </div>
                <p className="result-note">{result.model_note}</p>
                <button className="reset-button" onClick={resetForm} type="button"><RotateCcw size={15} /> Check another article</button>
              </div>
            )}
          </div>
        </section>

        {metrics && (
          <section className="metrics-panel" aria-label="Model evaluation metrics">
            <div className="metrics-heading"><div><span className="step-label">MODEL / EVALUATION</span><h2>Validation snapshot</h2></div><span className="metrics-tag">HELD-OUT TEST SET</span></div>
            <div className="metrics-grid">
              <div><span>ACCURACY</span><strong>{percent(metrics.accuracy)}</strong></div>
              <div><span>FAKE PRECISION</span><strong>{percent(metrics.precision_fake)}</strong></div>
              <div><span>FAKE RECALL</span><strong>{percent(metrics.recall_fake)}</strong></div>
              <div><span>FAKE F1-SCORE</span><strong>{percent(metrics.f1_fake)}</strong></div>
            </div>
            <p>These figures describe performance on a random split of the historical training dataset. They do not measure reliability on current news or unseen publishers.</p>
          </section>
        )}

        <section className="method-note"><div className="method-icon"><Info size={18} /></div><div><strong>Understand the result</strong><p>This is a machine-learning estimate, not a fact-check. The model was trained on historical headline and article text; it does not verify claims against live sources.</p></div></section>
      </main>

      <footer><span>VERITY <span className="brand-dot">·</span> FAKE NEWS PREDICTION</span><span>Built with React + FastAPI</span></footer>
    </div>
  )
}

export default App
