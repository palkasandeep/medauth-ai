import React, { useEffect, useState } from "react";
import { Send, Trash2, BookOpen } from "lucide-react";
import { fetchSavedReports, sendCaseChat } from "../api/client";
import { ChatMessage, ReportSummary } from "../types/medauth";

export const CaseAssistantPage: React.FC<{ refreshTrigger: number }> = ({
  refreshTrigger,
}) => {
  const [reports, setReports] = useState<ReportSummary[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string>("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchSavedReports()
      .then((list) => {
        setReports(list);
        if (list.length > 0 && !selectedPatientId) {
          setSelectedPatientId(list[0].patient_id);
        }
      })
      .catch(() => {});
  }, [refreshTrigger]);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || !selectedPatientId) return;

    const userMsg: ChatMessage = {
      id: `u-${Date.now()}`,
      role: "user",
      content: query.trim(),
    };
    const nextHistory = [...messages, userMsg];
    setMessages(nextHistory);
    setQuery("");
    setLoading(true);

    try {
      const res = await sendCaseChat({
        patientId: selectedPatientId,
        query: userMsg.content,
        history: nextHistory.map((m) => ({ role: m.role, content: m.content })),
      });

      const assistantMsg: ChatMessage = {
        id: `a-${Date.now()}`,
        role: "assistant",
        content: res.chat_answer ?? res.error ?? "No response available.",
        citations: res.citations ?? [],
        reflection: res.chat_reflection,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          role: "assistant",
          content:
            err instanceof Error ? err.message : "Failed to query Case Assistant.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 h-full flex flex-col">
      <header className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">
            Case Assistant (RAG)
          </h2>
          <p className="text-sm text-slate-600">
            Ask grounded clinical questions scoped strictly to a single finished
            patient report.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <select
            value={selectedPatientId}
            onChange={(e) => {
              setSelectedPatientId(e.target.value);
              setMessages([]);
            }}
            className="rounded-lg border border-slate-300 px-3 py-2 text-sm bg-white min-w-[240px]"
          >
            {reports.length === 0 ? (
              <option value="">No saved reports yet</option>
            ) : (
              reports.map((r) => (
                <option key={r.report_id} value={r.patient_id}>
                  {r.patient_id} — {r.patient_name} ({r.decision.decision})
                </option>
              ))
            )}
          </select>

          <button
            onClick={() => setMessages([])}
            className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg border border-slate-300 text-sm text-slate-700 hover:bg-slate-100"
          >
            <Trash2 className="w-4 h-4" />
            Clear Chat
          </button>
        </div>
      </header>

      <div className="flex-1 bg-white rounded-xl border border-slate-200 shadow-sm p-6 flex flex-col min-h-[460px]">
        <div className="flex-1 space-y-4 overflow-y-auto mb-4">
          {messages.length === 0 ? (
            <div className="text-center text-slate-400 py-16 text-sm">
              Select a saved patient report above and ask about eligibility,
              policy rules, risk indicators, or final determination.
            </div>
          ) : (
            messages.map((msg) => (
              <div
                key={msg.id}
                className={`p-4 rounded-xl max-w-3xl ${
                  msg.role === "user"
                    ? "ml-auto bg-medcore-600 text-white"
                    : "bg-slate-50 border border-slate-200 text-slate-800"
                }`}
              >
                <p className="text-sm whitespace-pre-wrap">{msg.content}</p>

                {msg.citations && msg.citations.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-slate-200 space-y-1.5">
                    <p className="text-xs font-semibold text-slate-500 flex items-center gap-1">
                      <BookOpen className="w-3.5 h-3.5" />
                      Report Chunks Cited
                    </p>
                    {msg.citations.map((c) => (
                      <div
                        key={c.chunk_id}
                        className="text-xs bg-white border border-slate-200 rounded p-2 text-slate-600"
                      >
                        <span className="font-mono font-semibold text-medcore-700">
                          [{c.chunk_id}]
                        </span>{" "}
                        {c.text}
                      </div>
                    ))}
                  </div>
                )}

                {msg.reflection && (
                  <div className="mt-3 pt-2 border-t border-slate-200 flex flex-wrap gap-2 text-[11px]">
                    <span className="px-2 py-0.5 rounded bg-slate-200 text-slate-700">
                      Answer Rel: {(msg.reflection.answer_relevance * 100).toFixed(0)}%
                    </span>
                    <span className="px-2 py-0.5 rounded bg-slate-200 text-slate-700">
                      Groundedness: {(msg.reflection.groundedness * 100).toFixed(0)}%
                    </span>
                    <span className="px-2 py-0.5 rounded bg-slate-200 text-slate-700">
                      Context Rel: {(msg.reflection.context_relevance * 100).toFixed(0)}%
                    </span>
                    <span
                      className={`px-2 py-0.5 rounded font-semibold ${
                        msg.reflection.passed
                          ? "bg-emerald-100 text-emerald-800"
                          : "bg-rose-100 text-rose-800"
                      }`}
                    >
                      RAG Triad: {msg.reflection.passed ? "PASS" : "REVIEW"}
                    </span>
                  </div>
                )}
              </div>
            ))
          )}
        </div>

        <form onSubmit={handleSend} className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question about this patient's PA report..."
            disabled={!selectedPatientId || loading}
            className="flex-1 rounded-lg border border-slate-300 px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-medcore-600"
          />
          <button
            type="submit"
            disabled={!selectedPatientId || loading || !query.trim()}
            className="inline-flex items-center gap-2 bg-medcore-600 hover:bg-medcore-700 disabled:opacity-50 text-white px-5 py-2.5 rounded-lg text-sm font-medium"
          >
            <Send className="w-4 h-4" />
            {loading ? "Thinking..." : "Send"}
          </button>
        </form>
      </div>
    </div>
  );
};
