import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import {
  addKnowledgeBaseDocuments,
  askKnowledgeBase,
  deleteKnowledgeBase,
  deleteKnowledgeBaseDocument,
  getKnowledgeBase,
  listTraces,
  uploadKnowledgeBaseDocuments,
} from "../api";
import { STATUS_LABELS, formatDate } from "../i18n";
import {
  EmptyState,
  ErrorMessage,
  LoadingIndicator,
  MetricChip,
  StatusPill,
  SuccessMessage,
} from "../components";
import { isTauriRuntime, pickDesktopDocuments } from "../runtime";

const ACCEPT_EXTENSIONS = ".md,.markdown,.txt,.rst,.json,.yaml,.yml,.html,.htm,.pdf";

export function KnowledgeBaseDetailPage() {
  const { kbId } = useParams();
  const navigate = useNavigate();
  const [knowledgeBase, setKnowledgeBase] = useState(null);
  const [traces, setTraces] = useState([]);
  const [question, setQuestion] = useState("");
  const [answerResult, setAnswerResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [asking, setAsking] = useState(false);
  const [updatingDocuments, setUpdatingDocuments] = useState(false);
  const [deletingKnowledgeBase, setDeletingKnowledgeBase] = useState(false);
  const [error, setError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [desktopMode] = useState(() => isTauriRuntime());
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [selectedPaths, setSelectedPaths] = useState([]);
  const [fileInputKey, setFileInputKey] = useState(0);
  const answerRef = useRef(null);
  const documents = knowledgeBase?.documents ?? [];

  useEffect(() => {
    let active = true;

    async function loadWorkspace() {
      setLoading(true);
      setError("");
      setSuccessMessage("");
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

  useEffect(() => {
    if (answerResult && answerRef.current) {
      answerRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [answerResult]);

  const handleQuestionChange = useCallback((event) => {
    setQuestion(event.target.value);
  }, []);

  async function handleAsk(event) {
    event.preventDefault();
    if (!question.trim()) {
      setError("请输入问题后再提问。");
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

  async function handleChooseDesktopDocuments() {
    setError("");
    try {
      const paths = await pickDesktopDocuments();
      setSelectedPaths(paths);
    } catch (pickError) {
      setError(
        pickError instanceof Error
          ? pickError.message
          : "无法打开本地文件选择器。"
      );
    }
  }

  async function handleAddDocuments(event) {
    event.preventDefault();
    if (!knowledgeBase) {
      return;
    }
    if (desktopMode && selectedPaths.length === 0) {
      setError("请至少选择一个本地文档路径。");
      setSuccessMessage("");
      return;
    }
    if (!desktopMode && selectedFiles.length === 0) {
      setError("请至少选择一个本地文档。");
      setSuccessMessage("");
      return;
    }

    setUpdatingDocuments(true);
    setError("");
    setSuccessMessage("");
    try {
      const payload = desktopMode
        ? await addKnowledgeBaseDocuments(kbId, { paths: selectedPaths })
        : await uploadKnowledgeBaseDocuments(kbId, { files: selectedFiles });
      setKnowledgeBase(payload);
      setSelectedFiles([]);
      setSelectedPaths([]);
      setFileInputKey((current) => current + 1);
      setSuccessMessage("文档已添加到当前知识库。");
    } catch (submitError) {
      setError(submitError.message);
    } finally {
      setUpdatingDocuments(false);
    }
  }

  async function handleDeleteDocument(documentId) {
    if (!knowledgeBase) {
      return;
    }

    setUpdatingDocuments(true);
    setError("");
    setSuccessMessage("");
    try {
      const payload = await deleteKnowledgeBaseDocument(kbId, documentId);
      setKnowledgeBase(payload);
      setSuccessMessage("文档已从当前知识库移除。");
    } catch (deleteError) {
      setError(deleteError.message);
    } finally {
      setUpdatingDocuments(false);
    }
  }

  async function handleDeleteKnowledgeBase() {
    if (!knowledgeBase) {
      return;
    }

    setDeletingKnowledgeBase(true);
    setError("");
    setSuccessMessage("");
    try {
      await deleteKnowledgeBase(kbId);
      await navigate("/");
    } catch (deleteError) {
      setError(deleteError.message);
    } finally {
      setDeletingKnowledgeBase(false);
    }
  }

  return (
    <main className="page-shell" id="main-content">
      <section className="hero-panel hero-panel-compact" aria-labelledby="detail-title">
        <Link className="back-link" to="/">
          ← 返回知识库列表
        </Link>
        <div className="eyebrow">知识工作区</div>
        <h1 id="detail-title">
          {knowledgeBase ? knowledgeBase.name : "正在加载工作区..."}
        </h1>
        <p className="hero-copy">
          基于知识库内容提问，查看引用来源，追踪最近的活动记录。
        </p>
      </section>

      <ErrorMessage>{error}</ErrorMessage>
      <SuccessMessage>{successMessage}</SuccessMessage>

      <section className="layout-grid detail-grid" aria-label="知识库详情区域">
        <article className="panel detail-column" aria-labelledby="metadata-heading">
          {loading ? (
            <LoadingIndicator text="正在加载知识库..." />
          ) : null}

          {knowledgeBase ? (
            <>
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">元数据</p>
                  <h2 id="metadata-heading">{knowledgeBase.name}</h2>
                </div>
                <div className="panel-actions">
                  <StatusPill
                    status={
                      STATUS_LABELS[knowledgeBase.status] || knowledgeBase.status
                    }
                  />
                  <button
                    className="secondary-button"
                    disabled={deletingKnowledgeBase}
                    onClick={handleDeleteKnowledgeBase}
                    type="button"
                  >
                    {deletingKnowledgeBase ? "删除中…" : "删除知识库"}
                  </button>
                </div>
              </div>

              <dl className="stat-grid stat-grid-expanded">
                <div>
                  <dt>文档数</dt>
                  <dd>{knowledgeBase.document_count}</dd>
                </div>
                <div>
                  <dt>分块数</dt>
                  <dd>{knowledgeBase.chunk_count}</dd>
                </div>
                <div>
                  <dt>更新时间</dt>
                  <dd>{formatDate(knowledgeBase.updated_at)}</dd>
                </div>
              </dl>

              <div className="source-block">
                <div className="panel-heading">
                  <div>
                    <h3 id="sources-heading">文档管理</h3>
                    <p className="message muted-message compact-message">
                      逐个文档增量入库，避免一次性大批量写入。
                    </p>
                  </div>
                  <MetricChip>{documents.length} 份文档</MetricChip>
                </div>

                <form className="stack" onSubmit={handleAddDocuments} noValidate>
                  {desktopMode ? (
                    <>
                      <div className="field">
                        <span id="kb-documents-label">添加本地文档</span>
                        <button
                          aria-labelledby="kb-documents-label"
                          className="secondary-button"
                          onClick={handleChooseDesktopDocuments}
                          type="button"
                        >
                          选择文档
                        </button>
                      </div>

                      {selectedPaths.length ? (
                        <ul className="file-list" aria-label="待添加路径">
                          {selectedPaths.map((path) => (
                            <li className="file-chip" key={path}>
                              {path}
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p className="message muted-message compact-message">
                          选择一个或多个文档追加到当前知识库。
                        </p>
                      )}
                    </>
                  ) : (
                    <>
                      <label className="field" htmlFor="kb-detail-files">
                        <span>添加本地文档</span>
                        <input
                          id="kb-detail-files"
                          key={fileInputKey}
                          accept={ACCEPT_EXTENSIONS}
                          multiple
                          name="files"
                          type="file"
                          onChange={(event) =>
                            setSelectedFiles(Array.from(event.target.files ?? []))
                          }
                        />
                      </label>

                      {selectedFiles.length ? (
                        <ul className="file-list" aria-label="待添加文件">
                          {selectedFiles.map((file) => (
                            <li className="file-chip" key={`${file.name}-${file.size}`}>
                              {file.name}
                            </li>
                          ))}
                        </ul>
                      ) : null}
                    </>
                  )}

                  <button
                    aria-busy={updatingDocuments}
                    className="primary-button"
                    disabled={updatingDocuments}
                    type="submit"
                  >
                    {updatingDocuments ? "处理中…" : "添加文档"}
                  </button>
                </form>

                {documents.length ? (
                  <ul className="document-list" aria-labelledby="sources-heading">
                    {documents.map((document) => (
                      <li className="document-row" key={document.document_id}>
                        <div className="document-copy">
                          <strong>{document.name}</strong>
                          <p>{document.source_path}</p>
                          <span>
                            {document.chunk_count} chunks · {document.size_bytes} bytes
                          </span>
                        </div>
                        <button
                          className="secondary-button"
                          disabled={updatingDocuments}
                          onClick={() => handleDeleteDocument(document.document_id)}
                          type="button"
                        >
                          删除
                        </button>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="message muted-message">当前知识库还没有文档。</p>
                )}
              </div>
            </>
          ) : null}
        </article>

        <article className="panel detail-column" aria-labelledby="ask-heading">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">智能问答</p>
              <h2 id="ask-heading">向此知识库提问</h2>
            </div>
            <MetricChip>追踪记录</MetricChip>
          </div>

          <form className="stack" onSubmit={handleAsk} noValidate>
            <label className="field" htmlFor="question-input">
              <span>问题</span>
              <textarea
                id="question-input"
                name="question"
                rows="5"
                placeholder="上次发布有哪些变更？"
                value={question}
                onChange={handleQuestionChange}
              />
            </label>
            <button
              aria-busy={asking}
              className="primary-button"
              disabled={asking}
              type="submit"
            >
              {asking ? "提问中…" : "提问"}
            </button>
          </form>

          {answerResult ? (
            <section
              className="answer-panel"
              aria-labelledby="answer-heading"
              ref={answerRef}
            >
              <div className="answer-panel-header">
                <h3 id="answer-heading">最新回答</h3>
                <StatusPill
                  status={answerResult.answered ? "已回答" : "已拒答"}
                />
              </div>
              <p className="answer-copy">{answerResult.answer}</p>
              {answerResult.reason ? (
                <p className="message muted-message">
                  原因：{answerResult.reason}
                </p>
              ) : null}
              {answerResult.citations.length ? (
                <ol className="citation-list" aria-label="引用列表">
                  {answerResult.citations.map((citation) => (
                    <li key={citation.chunk_id}>
                      <strong>[{citation.index}]</strong> {citation.source}
                      <p>{citation.snippet}</p>
                    </li>
                  ))}
                </ol>
              ) : (
                <p className="message muted-message">无引用信息。</p>
              )}
              {answerResult.trace_id ? (
                <p className="trace-line">追踪 ID：{answerResult.trace_id}</p>
              ) : null}
            </section>
          ) : null}
        </article>
      </section>

      <section className="panel" aria-labelledby="traces-heading">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">最近活动</p>
            <h2 id="traces-heading">最近追踪记录</h2>
          </div>
          <MetricChip>{traces.length} 条</MetricChip>
        </div>

        {traces.length === 0 ? (
          <EmptyState>
            <p className="message muted-message">
              暂无追踪记录，提问后将自动生成。
            </p>
          </EmptyState>
        ) : (
          <div className="trace-grid" role="list">
            {traces.map((trace) => (
              <article className="trace-card" key={trace.trace_id} role="listitem">
                <div className="trace-card-header">
                  <time className="trace-date" dateTime={trace.created_at}>
                    {formatDate(trace.created_at)}
                  </time>
                  <StatusPill status={trace.answered ? "已回答" : "已拒答"} />
                </div>
                <h3>{trace.question}</h3>
                <p>{trace.answer || "未捕获回答文本。"}</p>
              </article>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
