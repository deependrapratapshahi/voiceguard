import { useEffect, useRef, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { api } from "../services/api";
import { fileToWavChunks } from "../services/wavEncoder";
import RiskBadge from "../components/RiskBadge";
import { useCallStream } from "../hooks/useCallStream";

const CHUNK_SECONDS = 2.0;

interface ContextControls {
  bypass_requested: boolean;
  privileged_operation: boolean;
  urgency_indicated: boolean;
  caller_is_registered_contact: boolean;
  is_new_device: boolean;
  caller_reputation_score: number;
  transaction_value: number;
  historical_fraud_flags: number;
}

const DEFAULT_CONTEXT: ContextControls = {
  bypass_requested: false,
  privileged_operation: false,
  urgency_indicated: false,
  caller_is_registered_contact: true,
  is_new_device: false,
  caller_reputation_score: 1.0,
  transaction_value: 0,
  historical_fraud_flags: 0,
};

export default function LiveCallMonitoring() {
  const [callId, setCallId] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [streamStatus, setStreamStatus] = useState("");
  const [contextControls, setContextControls] = useState<ContextControls>(DEFAULT_CONTEXT);
  const streamStartedRef = useRef(false);

  const { results, connected, speakerReferenceLoaded, lastError, sendAudioChunk, sendContextUpdate } =
    useCallStream(callId);

  const startSimulatedCall = async () => {
    streamStartedRef.current = false;
    const call = await api.createCall({ caller_label: selectedFile?.name ?? "Demo Caller", is_demo: true });
    setCallId(call.call_id);
  };

  const applyContextControls = () => {
    sendContextUpdate(contextControls as unknown as Record<string, unknown>);
  };

  // Once the WebSocket is actually open, stream the selected file's
  // audio chunks one by one, paced roughly like a real live call.
  useEffect(() => {
    if (!connected || !selectedFile || streamStartedRef.current) return;
    streamStartedRef.current = true;

    (async () => {
      setStreaming(true);
      setStreamStatus("Decoding audio...");
      try {
        const chunks = await fileToWavChunks(selectedFile, CHUNK_SECONDS);
        setStreamStatus(`Streaming ${chunks.length} chunk(s)...`);
        for (let i = 0; i < chunks.length; i++) {
          sendAudioChunk(chunks[i]);
          setStreamStatus(`Streaming chunk ${i + 1} of ${chunks.length}...`);
          await new Promise((resolve) => setTimeout(resolve, CHUNK_SECONDS * 1000));
        }
        setStreamStatus("Finished streaming demo audio.");
      } catch (err) {
        setStreamStatus("Could not decode this audio file. Try a WAV or MP3 file.");
      } finally {
        setStreaming(false);
      }
    })();
  }, [connected, selectedFile, sendAudioChunk]);

  const latest = results[results.length - 1];
  const chartData = results.map((r, i) => ({
    idx: i,
    risk: r.risk_score,
    synthetic: r.synthetic_probability * 100,
    speaker: r.speaker_similarity !== null ? r.speaker_similarity * 100 : null,
    prosody: r.prosody_anomaly * 100,
  }));

  return (
    <div className="space-y-6">
      <header className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-xl font-semibold text-slate-100">Live Call Monitoring</h1>
          <p className="text-sm text-slate-500">
            Simulates a live call by streaming an uploaded audio file over WebSocket, chunk by
            chunk, through the full detection pipeline &mdash; no telephony integration required.
          </p>
        </div>
      </header>

      <div className="bg-panel border border-panelBorder rounded-xl p-4 space-y-3">
        <h2 className="text-sm font-semibold text-slate-300">Start a Simulated Call</h2>
        <p className="text-xs text-slate-500">
          Use only synthetic or authorized demo/test audio &mdash; never a real person's voice
          without consent.
        </p>
        <div className="flex items-center gap-3 flex-wrap">
          <input
            type="file"
            accept="audio/*"
            disabled={streaming}
            onChange={(e) => setSelectedFile(e.target.files?.[0] ?? null)}
            className="text-sm text-slate-400"
          />
          <button
            onClick={startSimulatedCall}
            disabled={!selectedFile || streaming}
            className="bg-accent/10 text-accent border border-accent/40 px-4 py-2 rounded-lg text-sm hover:bg-accent/20 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Start Simulated Call
          </button>
        </div>
      </div>

      {callId && (
        <div className="bg-panel border border-panelBorder rounded-xl p-4 space-y-3">
          <h2 className="text-sm font-semibold text-slate-300">Simulate Fraud-Risk Context</h2>
          <p className="text-xs text-slate-500">
            Toggle these mid-call to see how the contextual risk factors change the score and
            recommended action in real time.
          </p>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm text-slate-300">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={contextControls.bypass_requested}
                onChange={(e) => setContextControls((c) => ({ ...c, bypass_requested: e.target.checked }))}
              />
              Bypass callback requested
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={contextControls.privileged_operation}
                onChange={(e) => setContextControls((c) => ({ ...c, privileged_operation: e.target.checked }))}
              />
              Privileged operation
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={contextControls.urgency_indicated}
                onChange={(e) => setContextControls((c) => ({ ...c, urgency_indicated: e.target.checked }))}
              />
              Urgency indicated
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={!contextControls.caller_is_registered_contact}
                onChange={(e) =>
                  setContextControls((c) => ({ ...c, caller_is_registered_contact: !e.target.checked }))
                }
              />
              Not a registered contact
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={contextControls.is_new_device}
                onChange={(e) => setContextControls((c) => ({ ...c, is_new_device: e.target.checked }))}
              />
              New/unrecognized device
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Historical fraud flags</span>
              <input
                type="number"
                min={0}
                max={10}
                value={contextControls.historical_fraud_flags}
                onChange={(e) =>
                  setContextControls((c) => ({ ...c, historical_fraud_flags: Number(e.target.value) }))
                }
                className="bg-bg border border-panelBorder rounded px-2 py-1 text-sm w-20"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Caller reputation (0 = bad, 1 = trusted)</span>
              <input
                type="number"
                min={0}
                max={1}
                step={0.1}
                value={contextControls.caller_reputation_score}
                onChange={(e) =>
                  setContextControls((c) => ({ ...c, caller_reputation_score: Number(e.target.value) }))
                }
                className="bg-bg border border-panelBorder rounded px-2 py-1 text-sm w-24"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Transaction value ($)</span>
              <input
                type="number"
                min={0}
                step={1000}
                value={contextControls.transaction_value}
                onChange={(e) =>
                  setContextControls((c) => ({ ...c, transaction_value: Number(e.target.value) }))
                }
                className="bg-bg border border-panelBorder rounded px-2 py-1 text-sm w-32"
              />
            </label>
          </div>
          <button
            onClick={applyContextControls}
            className="bg-accent/10 text-accent border border-accent/40 px-4 py-2 rounded-lg text-sm hover:bg-accent/20"
          >
            Apply to Call
          </button>
        </div>
      )}

      {callId && (
        <div className="bg-panel border border-panelBorder rounded-xl p-4 space-y-4">
          <div className="flex items-center gap-3 text-sm text-slate-400 flex-wrap">
            <span>Call: {callId}</span>
            <span className={connected ? "text-risklow" : "text-slate-500"}>
              {connected ? "\u25cf connected" : "\u25cb connecting..."}
            </span>
            <span className="text-slate-500">
              Speaker reference: {speakerReferenceLoaded ? "loaded" : "none registered"}
            </span>
            {streamStatus && <span className="text-accent">{streamStatus}</span>}
            {lastError && <span className="text-riskhigh">Last error: {lastError}</span>}
          </div>

          {latest && (
            <>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                <Metric label="Synthetic Probability" value={latest.synthetic_probability} isPercent />
                <Metric
                  label="Speaker Similarity"
                  value={latest.speaker_similarity}
                  isPercent
                  unavailableText="not verified"
                />
                <Metric label="Prosody Anomaly" value={latest.prosody_anomaly} isPercent />
                <Metric label="Context Risk" value={latest.context_risk} isPercent />
                <div className="bg-bg/50 border border-panelBorder rounded-lg p-3">
                  <div className="text-xs text-slate-500 mb-1">Risk Level</div>
                  <RiskBadge level={latest.risk_level} />
                </div>
                <Metric label="Overall Risk Score" value={latest.risk_score} isScore />
              </div>

              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <XAxis dataKey="idx" stroke="#475569" fontSize={11} />
                    <YAxis domain={[0, 100]} stroke="#475569" fontSize={11} />
                    <Tooltip contentStyle={{ background: "#111826", border: "1px solid #1f2937" }} />
                    <Legend wrapperStyle={{ fontSize: "11px" }} />
                    <Line type="monotone" dataKey="risk" name="Risk Score" stroke="#22d3ee" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="synthetic" name="Synthetic %" stroke="#f97316" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
                    <Line type="monotone" dataKey="speaker" name="Speaker Sim %" stroke="#a78bfa" strokeWidth={1.5} dot={false} strokeDasharray="4 2" connectNulls />
                    <Line type="monotone" dataKey="prosody" name="Prosody Anomaly %" stroke="#eab308" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="text-sm text-slate-400">
                <span className="font-medium text-slate-300">Reasons: </span>
                {latest.reasons.join(", ")}
              </div>

              <div className="text-sm text-slate-400">
                <span className="font-medium text-slate-300">Recommended Action: </span>
                <span className="text-accent">{latest.recommended_action}</span>
              </div>

              <div className="text-xs text-slate-500 flex gap-4 flex-wrap">
                <span>Latency &mdash; preprocessing: {latest.latency_ms.preprocessing}ms</span>
                <span>inference: {latest.latency_ms.inference}ms</span>
                <span>total: {latest.latency_ms.total}ms</span>
              </div>

              <p className="text-xs text-slate-500 italic border-t border-panelBorder pt-3">
                {latest.disclaimer}
              </p>
            </>
          )}

          {!latest && (
            <p className="text-sm text-slate-500">
              Waiting for the first analyzed chunk...
            </p>
          )}
        </div>
      )}

      {!callId && (
        <p className="text-sm text-slate-500">
          Select a demo audio file and click "Start Simulated Call" to see the full detection
          pipeline run in real time, exactly as it would on a live call.
        </p>
      )}
    </div>
  );
}

function Metric({
  label,
  value,
  isScore,
  isPercent,
  unavailableText = "n/a",
}: {
  label: string;
  value: number | null;
  isScore?: boolean;
  isPercent?: boolean;
  unavailableText?: string;
}) {
  return (
    <div className="bg-bg/50 border border-panelBorder rounded-lg p-3">
      <div className="text-xs text-slate-500 mb-1">{label}</div>
      <div className="text-lg font-semibold text-slate-100">
        {value === null
          ? unavailableText
          : isScore
          ? value.toFixed(1)
          : isPercent
          ? `${(value * 100).toFixed(1)}%`
          : value}
      </div>
    </div>
  );
}
