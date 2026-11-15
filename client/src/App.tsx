import React, { useState } from "react";
import { Sidebar } from "./components/Sidebar";
import { ProcessCasePage } from "./components/ProcessCasePage";
import { CaseAssistantPage } from "./components/CaseAssistantPage";
import { AuditLogSection } from "./components/AuditLogSection";

export function App() {
  const [activePage, setActivePage] = useState<"process" | "assistant">("process");
  const [auditOpen, setAuditOpen] = useState(true);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleRefresh = () => setRefreshTrigger((n) => n + 1);

  return (
    <div className="flex min-h-screen bg-slate-50">
      <Sidebar
        activePage={activePage}
        onSelectPage={setActivePage}
        onToggleAudit={() => setAuditOpen((v) => !v)}
        auditOpen={auditOpen}
      />

      <main className="flex-1 p-6 lg:p-8 max-w-6xl mx-auto w-full space-y-6">
        {activePage === "process" ? (
          <ProcessCasePage onCaseCompleted={handleRefresh} />
        ) : (
          <CaseAssistantPage refreshTrigger={refreshTrigger} />
        )}

        {auditOpen && <AuditLogSection refreshTrigger={refreshTrigger} />}
      </main>
    </div>
  );
}

export default App;
