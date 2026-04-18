import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import {
  createKnowledgeBase,
  listKnowledgeBases,
  uploadKnowledgeBase,
} from "../api";
import { isTauriRuntime, pickDesktopDocuments } from "../runtime";

const EMPTY_FORM = {
  name: "",
  description: "",
};

function formatDate(value) {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export function KnowledgeBaseListPage() {
  const [knowledgeBases, setKnowledgeBases] = useState([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [form, setForm] = useState(EMPTY_FORM);
  const [desktopMode] = useState(() => isTauriRuntime());
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [selectedPaths, setSelectedPaths] = useState([]);
  const [fileInputKey, setFileInputKey] = useState(0);

  useEffect(() => {
    let active = true;

    async function loadKnowledgeBases() {
      setLoading(true);
      setError("");
      try {
        const payload = await listKnowledgeBases();
        if (active) {
          setKnowledgeBases(payload);
        }
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

    void loadKnowledgeBases();
    return () => {
      active = false;
    };
  }, []);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!form.name.trim()) {
      setError("Provide a knowledge base name.");
      setSuccessMessage("");
      return;
    }
    if (desktopMode && selectedPaths.length === 0) {
      setError("Choose at least one local document path.");
      setSuccessMessage("");
      return;
    }
    if (!desktopMode && selectedFiles.length === 0) {
      setError("Choose at least one local document.");
      setSuccessMessage("");
      return;
    }

    setSubmitting(true);
    setError("");
    setSuccessMessage("");
    try {
      const created = desktopMode
        ? await createKnowledgeBase({
            name: form.name.trim(),
            description: form.description.trim(),
            paths: selectedPaths,
          })
        : await uploadKnowledgeBase({
            name: form.name.trim(),
            description: form.description.trim(),
            files: selectedFiles,
          });
      const refreshed = await listKnowledgeBases();
      setKnowledgeBases(refreshed);
      setForm(EMPTY_FORM);
      setSelectedFiles([]);
      setSelectedPaths([]);
      setFileInputKey((current) => current + 1);
      setSuccessMessage(`Imported ${created.name} successfully.`);
    } catch (submitError) {
      setError(submitError.message);
    } finally {
      setSubmitting(false);
    }
  }

  async function handleChooseDesktopDocuments() {
    setError("");
    const paths = await pickDesktopDocuments();
    setSelectedPaths(paths);
  }

  return (
    <main className="page-shell">
      <section className="hero-panel">
        <div className="eyebrow">Knowledge QA Console</div>
        <h1>Turn the MVP into an operator-friendly knowledge workspace.</h1>
        <p className="hero-copy">
          {desktopMode
            ? "Manage knowledge bases with native file selection and ingest documents directly from your machine."
            : "Manage knowledge bases, upload local documents from your machine, and move straight into evidence-backed QA once indexing completes."}
        </p>
      </section>

      <section className="layout-grid">
        <article className="panel panel-form">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Ingest</p>
              <h2>Create Knowledge Base</h2>
            </div>
            <span className="status-pill status-pending">Sync import</span>
          </div>

          <form className="stack" onSubmit={handleSubmit}>
            <label className="field">
              <span>Name</span>
              <input
                name="name"
                placeholder="Atlas Docs"
                value={form.name}
                onChange={(event) =>
                  setForm((current) => ({
                    ...current,
                    name: event.target.value,
                  }))
                }
              />
            </label>

            <label className="field">
              <span>Description</span>
              <input
                name="description"
                placeholder="Product and engineering knowledge"
                value={form.description}
                onChange={(event) =>
                  setForm((current) => ({
                    ...current,
                    description: event.target.value,
                  }))
                }
              />
            </label>

            {desktopMode ? (
              <>
                <div className="field">
                  <span>Local Documents</span>
                  <button
                    className="secondary-button"
                    onClick={handleChooseDesktopDocuments}
                    type="button"
                  >
                    Choose Documents
                  </button>
                </div>

                {selectedPaths.length ? (
                  <div className="file-list" aria-label="Selected paths">
                    {selectedPaths.map((path) => (
                      <span className="file-chip" key={path}>
                        {path}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="message muted-message compact-message">
                    Choose one or more local files using the native desktop dialog.
                  </p>
                )}
              </>
            ) : (
              <>
                <label className="field">
                  <span>Local Documents</span>
                  <input
                    key={fileInputKey}
                    multiple
                    name="files"
                    type="file"
                    onChange={(event) =>
                      setSelectedFiles(Array.from(event.target.files ?? []))
                    }
                  />
                </label>

                {selectedFiles.length ? (
                  <div className="file-list" aria-label="Selected files">
                    {selectedFiles.map((file) => (
                      <span className="file-chip" key={`${file.name}-${file.size}`}>
                        {file.name}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="message muted-message compact-message">
                    Choose one or more local files to ingest.
                  </p>
                )}
              </>
            )}

            {desktopMode ? (
              <p className="message muted-message compact-message">
                Desktop mode reads local file paths directly and expects the local
                Python API to be running on `127.0.0.1:8000`.
              </p>
            ) : null}

            <button className="primary-button" disabled={submitting} type="submit">
              {submitting ? "Importing..." : "Create Knowledge Base"}
            </button>
          </form>

          {error ? <p className="message error-message">{error}</p> : null}
          {successMessage ? (
            <p className="message success-message">{successMessage}</p>
          ) : null}
        </article>

        <article className="panel panel-list">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Inventory</p>
              <h2>Knowledge Bases</h2>
            </div>
            <span className="metric-chip">{knowledgeBases.length} total</span>
          </div>

          {loading ? <p className="message muted-message">Loading knowledge bases...</p> : null}

          {!loading && !error && knowledgeBases.length === 0 ? (
            <p className="message muted-message">
              No knowledge bases yet. Start by ingesting one on the left.
            </p>
          ) : null}

          <div className="card-list">
            {knowledgeBases.map((knowledgeBase) => (
              <Link
                className="kb-card"
                key={knowledgeBase.kb_id}
                to={`/knowledge-bases/${knowledgeBase.kb_id}`}
              >
                <div className="kb-card-header">
                  <div>
                    <h3>{knowledgeBase.name}</h3>
                    <p>{knowledgeBase.description || "No description provided."}</p>
                  </div>
                  <span className={`status-pill status-${knowledgeBase.status}`}>
                    {knowledgeBase.status}
                  </span>
                </div>
                <dl className="stat-grid">
                  <div>
                    <dt>Documents</dt>
                    <dd>{knowledgeBase.document_count}</dd>
                  </div>
                  <div>
                    <dt>Chunks</dt>
                    <dd>{knowledgeBase.chunk_count}</dd>
                  </div>
                </dl>
                <div className="card-footer">
                  <span>{formatDate(knowledgeBase.updated_at)}</span>
                  <span>Open workspace</span>
                </div>
              </Link>
            ))}
          </div>
        </article>
      </section>
    </main>
  );
}
