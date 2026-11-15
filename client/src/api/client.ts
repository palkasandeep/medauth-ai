import { ReportSummary } from "../types/medauth";

export async function fetchSamples(): Promise<
  Array<{ id: string; name: string; fileCount: number; files: string[] }>
> {
  const res = await fetch("/api/samples");
  const data = await res.json();
  return data.samples ?? [];
}

export async function fetchSavedReports(): Promise<ReportSummary[]> {
  const res = await fetch("/api/reports");
  const data = await res.json();
  return data.reports ?? [];
}

export async function fetchAuditLog(): Promise<string[]> {
  const res = await fetch("/api/audit-log");
  const data = await res.json();
  return data.entries ?? [];
}

export async function uploadBundleFiles(files: FileList): Promise<string[]> {
  const formData = new FormData();
  Array.from(files).forEach((file) => formData.append("files", file));
  const res = await fetch("/api/upload", { method: "POST", body: formData });
  const data = await res.json();
  return (data.uploaded ?? []).map((u: { path: string }) => u.path);
}

export async function runCasePipeline(payload: {
  sampleId?: string;
  filePaths?: string[];
}) {
  const res = await fetch("/api/cases/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return res.json();
}

export async function submitHumanReview(
  threadId: string,
  payload: {
    reviewerDecision: "APPROVED" | "DENIED" | "PENDED_ADDITIONAL_INFO";
    comments: string;
  }
) {
  const res = await fetch(`/api/cases/${encodeURIComponent(threadId)}/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return res.json();
}

export async function sendCaseChat(payload: {
  patientId: string;
  query: string;
  history: Array<{ role: string; content: string }>;
}) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return res.json();
}
