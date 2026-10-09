import React, { useState, useEffect } from 'react';
import { checkBackendHealth, askQuestion } from './api';

const EXAMPLE_QUESTIONS = [
  "How many days of annual leave do employees receive per year?",
  "What are the remote work rules regarding allowed days and equipment stipend?",
  "What is the domestic hotel reimbursement limit per night?",
  "How many weeks of paid parental leave are employees entitled to receive?",
  "What is the minimum character length required for company passwords?",
];

export default function App() {
  const [question, setQuestion] = useState('');
  const [enableTracing, setEnableTracing] = useState(true);
  const [loading, setLoading] = useState(false);
  const [healthStatus, setHealthStatus] = useState({ checked: false, online: false });
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showTrace, setShowTrace] = useState(false);

  // Poll backend health on initial load and every 15s
  useEffect(() => {
    let mounted = true;

    async function checkHealth() {
      const res = await checkBackendHealth();
      if (mounted) {
        setHealthStatus({ checked: true, online: res.online, detail: res.error });
      }
    }

    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!question.trim() || loading) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await askQuestion(question, enableTracing);
      setResult(response);
    } catch (err) {
      setError(err.message || 'An unexpected error occurred while querying PolicyPilot.');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      handleSubmit(e);
    }
  };

  const handleSelectExample = (ex) => {
    setQuestion(ex);
  };

  return (
    <div className="app-container">
      {/* Header with Title and Backend Health Indicator */}
      <header className="header">
        <div className="brand-wrapper">
          <div className="brand-icon" aria-hidden="true">🛡️</div>
          <div>
            <h1 className="brand-title">PolicyPilot</h1>
            <p className="brand-subtitle">
              Deterministic Enterprise Policy Copilot & Conflict Resolution Engine
            </p>
          </div>
        </div>

        <div className="health-badge" title={healthStatus.online ? "Backend connected" : "Backend unreachable"}>
          <span
            className={`health-dot ${
              !healthStatus.checked ? 'checking' : healthStatus.online ? 'online' : 'offline'
            }`}
          />
          <span>
            {!healthStatus.checked
              ? 'Checking API...'
              : healthStatus.online
              ? 'API Online (FastAPI)'
              : 'API Offline'}
          </span>
        </div>
      </header>

      {/* Query Card */}
      <section className="query-card">
        <form onSubmit={handleSubmit} className="query-input-wrapper">
          <label htmlFor="policy-query-input" style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
            Ask a Policy or Compliance Question
          </label>
          <textarea
            id="policy-query-input"
            className="query-textarea"
            placeholder="e.g., How many days of annual leave can be carried forward?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
            rows={3}
          />

          <div className="query-actions">
            <label className="query-options">
              <input
                type="checkbox"
                checked={enableTracing}
                onChange={(e) => setEnableTracing(e.target.checked)}
              />
              Capture stage diagnostics & tracing
            </label>

            <button
              type="submit"
              className="btn-primary"
              disabled={loading || !question.trim()}
              id="ask-submit-button"
            >
              {loading ? (
                <>
                  <span className="spinner" style={{ width: 14, height: 14, borderWidth: 2, margin: 0 }} />
                  Synthesizing...
                </>
              ) : (
                'Ask PolicyPilot'
              )}
            </button>
          </div>
        </form>

        {/* Synthetic Document Example Chips */}
        <div className="examples-section">
          <div className="examples-title">Example Test Questions</div>
          <div className="example-chips">
            {EXAMPLE_QUESTIONS.map((ex, idx) => (
              <button
                key={idx}
                type="button"
                className="example-chip"
                onClick={() => handleSelectExample(ex)}
                disabled={loading}
              >
                {ex}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Error Banner */}
      {error && (
        <div className="error-box" role="alert">
          <strong>Query Error:</strong> {error}
        </div>
      )}

      {/* Loading State Indicator */}
      {loading && (
        <div className="loading-state" role="status">
          <div className="spinner" />
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9375rem' }}>
            Retrieving documents, verifying evidence, and checking cross-document consistency...
          </p>
        </div>
      )}

      {/* Results View */}
      {result && !loading && (
        <section className="result-card">
          <div className="result-header">
            <div className="result-question">“{result.question}”</div>
            <span className={`status-pill ${result.status || 'grounded'}`}>
              ● {result.status}
            </span>
          </div>

          <div className="result-answer">
            {result.answer}
          </div>

          {/* Citations Section */}
          {result.citations && result.citations.length > 0 && (
            <div className="citations-wrapper">
              <div className="citations-title">
                Verified Citations & Sources ({result.citations.length})
              </div>
              <div className="citations-grid">
                {result.citations.map((c, idx) => (
                  <div key={idx} className="citation-card">
                    <div className="citation-badge">{c.citation}</div>
                    <div className="citation-doc">
                      {c.document_name || c.document_id || 'Policy Document'}
                    </div>
                    {c.page_number && (
                      <div className="citation-page">Page {c.page_number}</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Diagnostics / Tracing Accordion */}
          {result.trace && result.trace.enabled && (
            <div className="trace-wrapper">
              <button
                type="button"
                className="trace-toggle"
                onClick={() => setShowTrace(!showTrace)}
                aria-expanded={showTrace}
              >
                <span>Diagnostics & Execution Trace ({result.trace.total_duration_ms} ms)</span>
                <span>{showTrace ? '▲ Hide' : '▼ View Trace'}</span>
              </button>

              {showTrace && (
                <div className="trace-content">
                  <div className="trace-grid">
                    <div className="trace-item">
                      <div className="trace-label">Trace ID</div>
                      <div className="trace-value" style={{ fontSize: '0.75rem' }}>
                        {result.trace.trace_id}
                      </div>
                    </div>
                    <div className="trace-item">
                      <div className="trace-label">Total Duration</div>
                      <div className="trace-value">
                        {result.trace.total_duration_ms} ms
                      </div>
                    </div>
                    <div className="trace-item">
                      <div className="trace-label">Status Rationale</div>
                      <div className="trace-value" style={{ fontSize: '0.75rem', fontWeight: 400 }}>
                        {result.trace.status_reason || 'N/A'}
                      </div>
                    </div>
                  </div>

                  {result.trace.stages && result.trace.stages.length > 0 && (
                    <div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.5rem', textTransform: 'uppercase' }}>
                        Workflow Stages
                      </div>
                      <div className="trace-stages">
                        {result.trace.stages.map((stg, sIdx) => (
                          <div key={sIdx} className="stage-row">
                            <span className="stage-name">{stg.stage}</span>
                            <span className="stage-duration">{stg.duration_ms} ms</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </section>
      )}

      {/* Empty State when no query has been submitted yet */}
      {!result && !loading && !error && (
        <div className="empty-state">
          <p>Select an example question above or type your own policy query to begin.</p>
        </div>
      )}
    </div>
  );
}
