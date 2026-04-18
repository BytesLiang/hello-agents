import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { askKnowledgeBase, getKnowledgeBase, listTraces } from "../api";

function formatDate(value) {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export function KnowledgeBaseDetailPage() {
  const { kbId } = useParams();
  const [knowledgeBase, setKnowledgeBase] = useState(null);
  const [traces, setTraces] = useState([]);
  const [question, setQuestion] = useState("");
  const [answerResult, setAnswerResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadWorkspace() {
      setLoading(true);
      setError("");
      try {
        const [knowledgeBasePayload, tracePayload] = await Promise.all([
          getKnowledgeBase(kbId),
          listTraces(10),
        ]);
        if (!active) {
          return;
        }
        setKnowledgeBase(knowledgeBasePayload);
        setTraces(tracePayload);
      } catch (fetchError) {
        if (active) {
          setError(fetchError.message);
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    void loadWorkspace();
    return () => {
      active = false;
    };
  }, [kbId]);

  async function handleAsk(event) {
    event.preventDefault();
    if (!question.trim()) {
      setError("Provide a question before asking.");
      return;
    }

    setAsking(true);
    setError("");
    try {
      const payload = await askKnowledgeBase(kbId, question.trim());
      setAnswerResult(payload);
      const tracePayload = await listTraces(10);
      setTraces(tracePayload);
    } catch (submitError) {
      setError(submitError.message);
    } finally {
      setAsking(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="hero-panel hero-panel-compact">
        <Link className="back-link" to="/">
          Back to knowledge bases
        </Link>
        <div className="eyebrow">Knowledge Workspace</div>
        <h1>{knowledgeBase ? knowledgeBase.name : "Loading workspace..."}</h1>
        <p className="hero-copy">
          Ask grounded questions, inspect citations, and keep an eye on recent
          activity from the same management surface.
        </p>
      </section>

      {error ? <p className="message error-message">{error}</p> : null}

      <section className="layout-grid detail-grid">
        <article className="panel detail-column">
          {loading ? (
            <p className="message muted-message">Loading knowledge base...</p>
          ) : null}

          {knowledgeBase ? (
            <>
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">Metadata</p>
                  <h2>{knowledgeBase.name}</h2>
                </div>
                <span className={`status-pill status-${knowledgeBase.status}`}>
                  {knowledgeBase.status}
                </span>
              </div>

              <dl className="stat-grid stat-grid-expanded">
                <div>
                  <dt>Documents</dt>
                  <dd>{knowledgeBase.document_count}</dd>
                </div>
                <div>
                  <dt>Chunks</dt>
                  <dd>{knowledgeBase.chunk_count}</dd>
                </div>
                <div>
                  <dt>Updated</dt>
                  <dd>{formatDate(knowledgeBase.updated_at)}</dd>
                </div>
              </dl>

              <div className="source-block">
                <h3>Indexed Sources</h3>
                <ul className="source-list">
                  {knowledgeBase.source_paths.map((sourcePath) => (
                    <li key={sourcePath}>{sourcePath}</li>
                  ))}
                </ul>
              </div>
            </>
          ) : null}
        </article>

        <article className="panel detail-column">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Question Answering</p>
              <h2>Ask This Knowledge Base</h2>
            </div>
            <span className="metric-chip">Trace-backed</span>
          </div>

          <form className="stack" onSubmit={handleAsk}>
            <label className="field">
              <span>Question</span>
              <textarea
                name="question"
                rows="5"
                placeholder="What changed in the last release?"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
              />
            </label>
            <button className="primary-button" disabled={asking} type="submit">
              {asking ? "Asking..." : "Ask Knowledge Base"}
            </button>
          </form>

          {answerResult ? (
            <section className="answer-panel">
              <div className="answer-panel-header">
                <h3>Latest Answer</h3>
                <span
                  className={`status-pill ${
                    answerResult.answered ? "status-ready" : "status-failed"
                  }`}
                >
                  {answerResult.answered ? "answered" : "refused"}
                </span>
              </div>
              <p className="answer-copy">{answerResult.answer}</p>
              {answerResult.reason ? (
                <p className="message muted-message">
                  Reason: {answerResult.reason}
                </p>
              ) : null}
              {answerResult.citations.length ? (
                <ol className="citation-list">
                  {answerResult.citations.map((citation) => (
                    <li key={citation.chunk_id}>
                      <strong>[{citation.index}]</strong> {citation.source}
                      <p>{citation.snippet}</p>
                    </li>
                  ))}
                </ol>
              ) : (
                <p className="message muted-message">No citations returned.</p>
              )}
              {answerResult.trace_id ? (
                <p className="trace-line">Trace ID: {answerResult.trace_id}</p>
              ) : null}
            </section>
          ) : null}
        </article>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Recent Activity</p>
            <h2>Recent Traces</h2>
          </div>
          <span className="metric-chip">{traces.length} rows</span>
        </div>

        {traces.length === 0 ? (
          <p className="message muted-message">
            No recent traces yet. Ask the first question to generate one.
          </p>
        ) : (
          <div className="trace-grid">
            {traces.map((trace) => (
              <article className="trace-card" key={trace.trace_id}>
                <div className="trace-card-header">
                  <span className="trace-date">{formatDate(trace.created_at)}</span>
                  <span
                    className={`status-pill ${
                      trace.answered ? "status-ready" : "status-failed"
                    }`}
                  >
                    {trace.answered ? "answered" : "refused"}
                  </span>
                </div>
                <h3>{trace.question}</h3>
                <p>{trace.answer || "No answer text captured."}</p>
              </article>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
