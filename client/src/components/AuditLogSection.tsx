import React, { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { fetchAuditLog } from "../api/client";

export const AuditLogSection: React.FC<{ refreshTrigger: number }> = ({
  refreshTrigger,
}) => {
  const [entries, setEntries] = useState<string[]>([]);

  const load = () => {
    fetchAuditLog()
      .then(setEntries)
      .catch(() => {});
  };

  useEffect(() => {
    load();
  }, [refreshTrigger]);

  return (
    <section className="bg-slate-900 text-slate-100 rounded-xl p-5 border border-slate-700 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold tracking-wide uppercase text-sky-300">
          Immutable Compliance Audit Log
        </h3>
        <button
          onClick={load}
          className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Refresh
        </button>
      </div>
      <div className="max-h-56 overflow-y-auto font-mono text-xs space-y-1 bg-black/30 rounded-lg p-3">
        {entries.length === 0 ? (
          <p className="text-slate-400">No audit events recorded yet.</p>
        ) : (
          entries.map((line, idx) => (
            <div key={idx} className="text-slate-200 break-all">
              {line}
            </div>
          ))
        )}
      </div>
    </section>
  );
};
