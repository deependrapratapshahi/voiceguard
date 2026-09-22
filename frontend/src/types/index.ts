export type RiskLevel = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

export interface Call {
  call_id: string;
  caller_label: string | null;
  speaker_id: string | null;
  status: string;
  started_at: string;
  ended_at: string | null;
  is_demo: boolean;
}

export interface RiskSnapshot {
  call_id: string;
  timestamp: string;
  synthetic_probability: number | null;
  speaker_similarity: number | null;
  prosody_anomaly: number | null;
  risk_score: number;
  risk_level: RiskLevel;
  reasons: string[];
  recommended_action: string | null;
}

export interface Alert {
  alert_id: string;
  call_id: string;
  severity: RiskLevel;
  message: string;
  recommended_action: string;
  status: string;
  created_at: string;
}

export interface LatencyMs {
  preprocessing: number;
  inference: number;
  total: number;
}

export interface StreamResult {
  type: "result";
  call_id: string;
  timestamp: string;
  chunk_start_time: number;
  chunk_end_time: number;
  synthetic_probability: number;
  speaker_similarity: number | null;
  identity_match: boolean | null;
  prosody_anomaly: number;
  context_risk: number;
  risk_score: number;
  risk_level: RiskLevel;
  reasons: string[];
  recommended_action: string;
  alert: Alert | null;
  model_version: string;
  disclaimer: string;
  latency_ms: LatencyMs;
}

export interface ConnectedMessage {
  type: "connected";
  call_id: string;
  speaker_reference_loaded: boolean;
  disclaimer: string;
}

export interface ContextUpdatedMessage {
  type: "context_updated";
  call_id: string;
}

export interface ErrorMessage {
  type: "error";
  call_id?: string;
  error: string;
}

export type WSMessage = StreamResult | ConnectedMessage | ContextUpdatedMessage | ErrorMessage;

export interface HealthStatus {
  status: string;
  app: string;
  env: string;
  database: string;
  redis: string;
}

export interface ModelInfo {
  component: string;
  name: string;
  version: string;
  is_active: boolean;
  is_baseline: boolean;
}
