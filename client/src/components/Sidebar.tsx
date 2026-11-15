import React from "react";
import { ShieldCheck, FileSearch, MessageSquareText, Activity } from "lucide-react";

interface SidebarProps {
  activePage: "process" | "assistant";
  onSelectPage: (page: "process" | "assistant") => void;
  onToggleAudit: () => void;
  auditOpen: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activePage,
  onSelectPage,
  onToggleAudit,
  auditOpen,
}) => {
  return (
    <aside className="w-64 bg-medcore-800 text-white flex flex-col justify-between min-h-screen shrink-0">
      <div>
        <div className="px-6 py-5 border-b border-white/10 flex items-center gap-3">
          <div className="p-2 bg-sky-500/20 rounded-lg text-sky-300">
            <ShieldCheck className="w-6 h-6" />
          </div>
          <div>
            <h1 className="font-semibold text-lg tracking-tight">MedAuth AI</h1>
            <p className="text-xs text-sky-200/80">MedCore Clinical PA</p>
          </div>
        </div>

        <nav className="p-4 space-y-1.5">
          <button
            onClick={() => onSelectPage("process")}
            className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium transition ${
              activePage === "process"
                ? "bg-white text-medcore-800 shadow-sm"
                : "text-slate-200 hover:bg-white/10"
            }`}
          >
            <FileSearch className="w-4 h-4" />
            Process Case
          </button>

          <button
            onClick={() => onSelectPage("assistant")}
            className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium transition ${
              activePage === "assistant"
                ? "bg-white text-medcore-800 shadow-sm"
                : "text-slate-200 hover:bg-white/10"
            }`}
          >
            <MessageSquareText className="w-4 h-4" />
            Case Assistant
          </button>
        </nav>
      </div>

      <div className="p-4 border-t border-white/10">
        <button
          onClick={onToggleAudit}
          className={`w-full flex items-center justify-between px-4 py-2.5 rounded-lg text-sm font-medium transition ${
            auditOpen
              ? "bg-sky-500/20 text-sky-200 border border-sky-400/40"
              : "text-slate-200 hover:bg-white/10"
          }`}
        >
          <span className="flex items-center gap-2.5">
            <Activity className="w-4 h-4" />
            Audit Log
          </span>
          <span className="text-xs opacity-75">{auditOpen ? "Hide" : "View"}</span>
        </button>
      </div>
    </aside>
  );
};
