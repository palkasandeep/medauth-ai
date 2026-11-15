import React, { useEffect, useState } from "react";
import {
  Upload,
  Play,
  CheckCircle2,
  AlertTriangle,
  Clock,
  ShieldAlert,
  FileCheck2,
} from "lucide-react";
import {
  fetchSamples,
  runCasePipeline,
  submitHumanReview,
  uploadBundleFiles,
} from "../api/client";
import { NodeProgressItem, ReportSummary, WorkflowNodeName } from "../types/medauth";

const INITIAL_NODES: NodeProgressItem[] = [
  { node: "document", label: "Document & Translation", status: "pending" },
  { node: "completeness", label: "Completeness Check", status: "pending" },
  { node: "eligibility", label: "Payer Eligibility", status: "pending" },
  { node: "risk", label: "Verification, Risk & Peer Review", status: "pending" },
  { node: "retrieval", label: "Policy Rules Lookup", status: "pending" },
  { node: "decision", label: "Determination Synthesis", status: "pending" },
  { node: "reflection", label: "Deterministic Self-Check", status: "pending" },
  { node: "reporting", label: "Report & Audit Generation", status: "pending" },
  { node: "indexing", label: "RAG Report Indexing", status: "pending" },
  { node: "human_review", label: "Human Review Routing", status: "pending" },
];

export const ProcessCasePage: React.FC<{ onCaseCompleted: () => void }> = ({
  onCaseCompleted,
}) => {
  const [samples, setSamples] = useState<
    Array<{ id: string; name: string; fileCount: number }>
  >([]);
  const [selectedSample, setSelectedSample] = useState<string>("");
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);
  const [nodes, setNodes] = useState<NodeProgressItem[]>(INITIAL_NODES);
  const [running, setRunning] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [resultReport, setResultReport] = useState<ReportSummary | null>(null);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [reviewDecision, setReviewDecision] = useState<
    "APPROVED" | "DENIED" | "PENDED_ADDITIONAL_INFO"
  >("APPROVED");
  const [reviewComments, setReviewComments] = useState("");
  const [reviewStatusMsg, setReviewStatusMsg] = useState<string | null>(null);

  useEffect(() => {
    fetchSamples()
      .then((list) => {
        setSamples(list);
        if (list.length > 0) setSelectedSample(list[0].id);
      })
      .catch(() => {});
  }, []);

  const handleRun = async () => {
    setRunning(true);
    setErrorMsg(null);
    setReviewStatusMsg(null);
    setResultReport(null);
    setNodes(INITIAL_NODES.map((n) => ({ ...n, status: "running" })));

    try {
      let filePaths: string[] | undefined;
      if (selectedFiles && selectedFiles.length > 0) {
        filePaths = await uploadBundleFiles(selectedFiles);
      }

      const res = await runCasePipeline(
        filePaths ? { filePaths } : { sampleId: selectedSample }
      );

      if (res.error) {
        setErrorMsg(res.error);
        setNodes(INITIAL_NODES.map((n) => ({ ...n, status: "error" })));
      } else {
        if (res.nodeStatuses) {
          setNodes((prev) =>
            prev.map((n) => ({
              ...n,
              status:
                (res.nodeStatuses[n.node as WorkflowNodeName] as NodeProgressItem["status"]) ??
                (res.interrupted && n.node === "human_review" ? "running" : "completed"),
            }))
          );
        }
        setResultReport(res.report ?? null);
        setThreadId(res.threadId ?? null);
        onCaseCompleted();
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Pipeline execution failed.");
      setNodes(INITIAL_NODES.map((n) => ({ ...n, status: "error" })));
    } finally {
      setRunning(false);
    }
  };

  const handleSubmitReview = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!threadId) return;
    const res = await submitHumanReview(threadId, {
      reviewerDecision: reviewDecision,
      comments: reviewComments,
    });
    if (res.report) setResultReport(res.report);
    if (res.nodeStatuses) {
      setNodes((prev) =>
        prev.map((n) => ({
          ...n,
          status:
            (res.nodeStatuses[n.node as WorkflowNodeName] as NodeProgressItem["status"]) ??
            n.status,
        }))
      );
    }
    setReviewStatusMsg(`Reviewer decision recorded: ${reviewDecision}`);
    onCaseCompleted();
  };

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-2xl font-bold text-slate-900">
          Process Prior Authorization Case
        </h2>
        <p className="text-sm text-slate-600">
          Upload a PA document bundle or select a MedCore demo bundle to run the
          10-node agentic adjudication pipeline.
        </p>
      </header>

      <section className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Select Demo Case Bundle
            </label>
            <select
              value={selectedSample}
              onChange={(e) => {
                setSelectedSample(e.target.value);
                setSelectedFiles(null);
              }}
              className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-medcore-600"
            >
              {samples.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name} ({s.fileCount} files)
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Or Upload Custom Bundle (.txt, .pdf, .docx, .png, .json)
            </label>
            <label className="flex items-center justify-center gap-2 border-2 border-dashed border-slate-300 rounded-lg px-4 py-2 text-sm text-slate-600 hover:border-medcore-600 cursor-pointer">
              <Upload className="w-4 h-4" />
              <span>
                {selectedFiles && selectedFiles.length > 0
                  ? `${selectedFiles.length} file(s) selected`
                  : "Choose PA bundle files..."}
              </span>
              <input
                type="file"
                multiple
                className="hidden"
                onChange={(e) => setSelectedFiles(e.target.files)}
              />
            </label>
          </div>
        </div>

        <div className="flex items-center justify-between pt-2">
          <span className="text-xs text-slate-500">
            Active input:{" "}
            {selectedFiles && selectedFiles.length > 0
              ? `${selectedFiles.length} uploaded file(s)`
              : selectedSample || "None"}
          </span>
          <button
            onClick={handleRun}
            disabled={running || (!selectedSample && !selectedFiles?.length)}
            className="inline-flex items-center gap-2 bg-medcore-600 hover:bg-medcore-700 disabled:opacity-50 text-white px-5 py-2.5 rounded-lg text-sm font-medium transition"
          >
            <Play className="w-4 h-4" />
            {running ? "Running Pipeline..." : "Run PA Determination"}
          </button>
        </div>

        {errorMsg && (
          <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-900 text-sm">
            {errorMsg}
          </div>
        )}
      </section>

      <section className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
        <h3 className="text-sm font-semibold text-slate-800 uppercase tracking-wider mb-4">
          LangGraph Execution Tracker
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
          {nodes.map((item, idx) => (
            <div
              key={item.node}
              className={`rounded-lg border p-3 text-xs ${
                item.status === "completed"
                  ? "bg-emerald-50 border-emerald-200 text-emerald-900"
                  : item.status === "running"
                    ? "bg-sky-50 border-sky-300 text-sky-900"
                    : item.status === "skipped"
                      ? "bg-slate-50 border-slate-200 text-slate-400"
                      : item.status === "error"
                        ? "bg-rose-50 border-rose-200 text-rose-800"
                        : "bg-slate-50 border-slate-200 text-slate-600"
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="font-semibold">
                  {idx + 1}. {item.node}
                </span>
                {item.status === "completed" ? (
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                ) : (
                  <Clock className="w-3.5 h-3.5 opacity-60" />
                )}
              </div>
              <p className="truncate opacity-80">{item.label}</p>
            </div>
          ))}
        </div>
      </section>

      {resultReport && (
        <section className="space-y-6">
          {!resultReport.human_review_required ? (
            <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-4 flex items-center gap-3 text-emerald-900">
              <FileCheck2 className="w-5 h-5 text-emerald-600 shrink-0" />
              <div>
                <p className="font-semibold text-sm">
                  No manual approval required — Case Auto-Finalized
                </p>
                <p className="text-xs text-emerald-800">
                  High-confidence, low-risk approval met all MedCore policy rules and
                  logged to audit trail.
                </p>
              </div>
            </div>
          ) : resultReport.status === "complete" ? (
            <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-4 flex items-center gap-3 text-emerald-900">
              <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0" />
              <div>
                <p className="font-semibold text-sm">
                  Manual Review Complete — Reviewer Decision:{" "}
                  {resultReport.reviewer_decision}
                </p>
                {resultReport.reviewer_comments && (
                  <p className="text-xs text-emerald-800">
                    Comments: {resultReport.reviewer_comments}
                  </p>
                )}
              </div>
            </div>
          ) : (
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-6 space-y-4">
              <div className="flex items-center gap-2 text-amber-900">
                <AlertTriangle className="w-5 h-5 text-amber-600" />
                <h3 className="font-semibold">
                  Clinical Reviewer Sign-Off Required
                </h3>
              </div>
              <p className="text-sm text-amber-800">
                Routing reason: {resultReport.decision.routing.reason}
              </p>

              <form onSubmit={handleSubmitReview} className="space-y-3">
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  {(["APPROVED", "DENIED", "PENDED_ADDITIONAL_INFO"] as const).map(
                    (opt) => (
                      <label
                        key={opt}
                        className={`flex items-center gap-2 p-3 rounded-lg border cursor-pointer text-sm font-medium ${
                          reviewDecision === opt
                            ? "bg-white border-medcore-600 text-medcore-800"
                            : "bg-white/60 border-amber-200 text-slate-700"
                        }`}
                      >
                        <input
                          type="radio"
                          name="reviewDecision"
                          checked={reviewDecision === opt}
                          onChange={() => setReviewDecision(opt)}
                        />
                        {opt}
                      </label>
                    )
                  )}
                </div>

                <textarea
                  value={reviewComments}
                  onChange={(e) => setReviewComments(e.target.value)}
                  placeholder="Enter clinical reviewer rationale and sign-off notes..."
                  rows={3}
                  className="w-full rounded-lg border border-amber-300 bg-white p-3 text-sm"
                />

                <button
                  type="submit"
                  className="bg-medcore-700 hover:bg-medcore-800 text-white px-4 py-2 rounded-lg text-sm font-medium"
                >
                  Submit Reviewer Determination
                </button>

                {reviewStatusMsg && (
                  <p className="text-sm font-medium text-emerald-700">
                    {reviewStatusMsg}
                  </p>
                )}
              </form>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white rounded-xl border border-slate-200 p-5">
              <p className="text-xs uppercase text-slate-500 font-semibold">
                Determination
              </p>
              <p className="text-xl font-bold text-slate-900 mt-1">
                {resultReport.decision.decision}
              </p>
              <p className="text-sm text-slate-600 mt-1">
                Confidence:{" "}
                {(resultReport.decision.confidence_score * 100).toFixed(1)}%
              </p>
            </div>

            <div className="bg-white rounded-xl border border-slate-200 p-5">
              <p className="text-xs uppercase text-slate-500 font-semibold">
                Eligibility &amp; Cost Share
              </p>
              <p className="text-lg font-bold text-slate-900 mt-1">
                {resultReport.eligibility?.eligible ? "Eligible" : "Ineligible / Skipped"}
              </p>
              <p className="text-sm text-slate-600 mt-1">
                Coverage: {resultReport.eligibility?.coverage_percentage ?? 0}% |
                Patient Resp: $
                {resultReport.eligibility?.patient_responsibility?.toFixed(2) ??
                  "0.00"}
              </p>
            </div>

            <div className="bg-white rounded-xl border border-slate-200 p-5">
              <p className="text-xs uppercase text-slate-500 font-semibold">
                Risk &amp; Integrity
              </p>
              <p className="text-lg font-bold text-slate-900 mt-1 capitalize">
                {resultReport.risk?.risk_level ?? "N/A"} Risk
              </p>
              <p className="text-sm text-slate-600 mt-1">
                Fraud Indicators: {resultReport.risk?.fraud_indicators.length ?? 0}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-xl border border-slate-200 p-6 space-y-3">
              <h4 className="font-semibold text-slate-900">
                Policy Rule Evaluation
              </h4>
              <div className="space-y-2">
                {resultReport.rules_applied.map((rule) => (
                  <div
                    key={rule.rule_id}
                    className="p-3 rounded-lg border border-slate-100 bg-slate-50 flex items-start justify-between gap-3"
                  >
                    <div>
                      <p className="text-sm font-semibold text-slate-800">
                        {rule.rule_id}
                      </p>
                      <p className="text-xs text-slate-600 mt-0.5">{rule.reason}</p>
                    </div>
                    <span
                      className={`px-2 py-0.5 rounded text-xs font-semibold shrink-0 ${
                        rule.passed
                          ? "bg-emerald-100 text-emerald-800"
                          : "bg-rose-100 text-rose-800"
                      }`}
                    >
                      {rule.passed ? "PASS" : "FAIL"}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white rounded-xl border border-slate-200 p-6 space-y-3">
              <h4 className="font-semibold text-slate-900 flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 text-slate-600" />
                Risk &amp; Fraud Findings
              </h4>
              <p className="text-sm text-slate-700">
                {resultReport.risk?.reasoning ?? "No risk assessment recorded."}
              </p>
              {resultReport.risk?.fraud_indicators &&
                resultReport.risk.fraud_indicators.length > 0 && (
                  <ul className="list-disc pl-5 text-sm text-rose-700 space-y-1">
                    {resultReport.risk.fraud_indicators.map((ind, i) => (
                      <li key={i}>{ind}</li>
                    ))}
                  </ul>
                )}
            </div>
          </div>
        </section>
      )}
    </div>
  );
};
