/**
 * API client service for communicating with the PolicyPilot FastAPI backend.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Checks backend health status.
 */
export async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    if (!res.ok) {
      return { online: false, error: `HTTP ${res.status}` };
    }
    const data = await res.json();
    return { online: true, data };
  } catch (err) {
    return { online: false, error: err.message || 'Connection failed' };
  }
}

/**
 * Sends a question to the PolicyPilot LangGraph workflow backend.
 */
export async function askQuestion(question, enableTracing = true, traceId = null) {
  const payload = {
    question: question.trim(),
    enable_tracing: Boolean(enableTracing),
  };
  if (traceId) {
    payload.trace_id = traceId;
  }

  const res = await fetch(`${API_BASE_URL}/ask`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    let errorDetail = `Request failed with status ${res.status}`;
    try {
      const errJson = await res.json();
      if (errJson && errJson.detail) {
        errorDetail = errJson.detail;
      }
    } catch {
      // fallback to status code
    }
    throw new Error(errorDetail);
  }

  return await res.json();
}
