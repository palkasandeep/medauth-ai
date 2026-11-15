export type WorkflowNodeName =
  | "document"
  | "completeness"
  | "eligibility"
  | "risk"
  | "retrieval"
  | "decision"
  | "reflection"
  | "reporting"
  | "indexing"
  | "human_review";

export interface NodeProgressItem {
  node: WorkflowNodeName;
  label: string;
  status: "pending" | "running" | "completed" | "skipped" | "error";
}

export interface RuleResult {
  rule_id: string;
  rule_text?: string;
  passed: boolean;
  reason: string;
}

export interface ReportSummary {
  report_id: string;
  case_id: string;
  thread_id?: string;
  patient_id: string;
  patient_name: string;
  requested_service: string;
  generated_at: string;
  human_review_required: boolean;
  status: string;
  eligibility?: {
    eligible: boolean;
    eligibility_reason: string;
    coverage_percentage: number;
    patient_responsibility: number;
    plan_name: string;
  };
  risk?: {
    risk_level: "low" | "medium" | "high";
    risk_score: number;
    crosswalk_match: boolean;
    is_duplicate: boolean;
    provider_verified: boolean;
    fraud_indicators: string[];
    requires_peer_review: boolean;
    reasoning: string;
  };
  rules_applied: RuleResult[];
  decision: {
    decision: string;
    confidence_score: number;
    key_findings: string[];
    rule_results: RuleResult[];
    reasoning: string;
    routing: {
      escalate_to_human: boolean;
      reason: string;
    };
  };
  reviewer_decision?: string | null;
  reviewer_comments?: string | null;
}

export interface RagTriad {
  answer_relevance: number;
  groundedness: number;
  context_relevance: number;
  overall_score: number;
  passed: boolean;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Array<{ chunk_id: string; text: string; score?: number }>;
  reflection?: RagTriad;
}
