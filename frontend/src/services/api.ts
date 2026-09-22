import axios from "axios";
import type { Alert, Call, HealthStatus, ModelInfo, RiskSnapshot } from "../types";

const client = axios.create({ baseURL: "/api/v1" });

// If the backend is unreachable (wrong proxy, CORS, etc.) some servers
// respond with an HTML error page instead of JSON. Guard against that
// here so a malformed response degrades to an empty list instead of
// crashing the page with "x.filter is not a function".
function asArray<T>(data: unknown): T[] {
  return Array.isArray(data) ? (data as T[]) : [];
}

export const api = {
  health: () => client.get<HealthStatus>("/health").then((r) => r.data),
  models: () => client.get<ModelInfo[]>("/models").then((r) => asArray<ModelInfo>(r.data)),

  listCalls: () => client.get<Call[]>("/calls").then((r) => asArray<Call>(r.data)),
  createCall: (payload: { caller_label?: string; speaker_id?: string; is_demo?: boolean }) =>
    client.post<Call>("/calls", payload).then((r) => r.data),
  getCall: (callId: string) => client.get<Call>(`/calls/${callId}`).then((r) => r.data),
  getCallRisk: (callId: string) =>
    client.get<RiskSnapshot[]>(`/calls/${callId}/risk`).then((r) => asArray<RiskSnapshot>(r.data)),

  listAlerts: () => client.get<Alert[]>("/alerts").then((r) => asArray<Alert>(r.data)),

  analyzeAudio: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return client.post("/detection/analyze", form, {
      headers: { "Content-Type": "multipart/form-data" },
    }).then((r) => r.data);
  },

  registerSpeaker: (speakerId: string, displayName: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return client
      .post(`/speakers?speaker_id=${encodeURIComponent(speakerId)}&display_name=${encodeURIComponent(displayName)}`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      .then((r) => r.data);
  },

  verifySpeaker: (speakerId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return client
      .post(`/speakers/${encodeURIComponent(speakerId)}/verify`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      .then((r) => r.data);
  },
};
