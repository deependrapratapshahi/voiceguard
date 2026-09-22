import { useState } from "react";
import { api } from "../services/api";

export default function SpeakerRegistry() {
  const [speakerId, setSpeakerId] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [refFile, setRefFile] = useState<File | null>(null);
  const [testFile, setTestFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [status, setStatus] = useState("");

  const register = async () => {
    if (!refFile || !speakerId || !displayName) return;
    setStatus("Registering...");
    try {
      await api.registerSpeaker(speakerId, displayName, refFile);
      setStatus("Speaker profile created.");
    } catch (e) {
      setStatus("Failed to register speaker.");
    }
  };

  const verify = async () => {
    if (!testFile || !speakerId) return;
    setStatus("Verifying...");
    try {
      const res = await api.verifySpeaker(speakerId, testFile);
      setResult(res);
      setStatus("");
    } catch (e) {
      setStatus("Verification failed.");
    }
  };

  return (
    <div className="space-y-6 max-w-2xl">
      <header>
        <h1 className="text-xl font-semibold text-slate-100">Speaker Registry</h1>
        <p className="text-sm text-slate-500">
          Register a reference voice (demo/authorized identities only), then test verification
          against new audio samples.
        </p>
      </header>

      <div className="bg-panel border border-panelBorder rounded-xl p-4 space-y-3">
        <h2 className="text-sm font-semibold text-slate-300">1. Create Speaker Profile</h2>
        <input
          className="w-full bg-bg border border-panelBorder rounded-lg px-3 py-2 text-sm"
          placeholder="Speaker ID (e.g. demo-alice)"
          value={speakerId}
          onChange={(e) => setSpeakerId(e.target.value)}
        />
        <input
          className="w-full bg-bg border border-panelBorder rounded-lg px-3 py-2 text-sm"
          placeholder="Display Name"
          value={displayName}
          onChange={(e) => setDisplayName(e.target.value)}
        />
        <input type="file" accept="audio/*" onChange={(e) => setRefFile(e.target.files?.[0] ?? null)} />
        <button
          onClick={register}
          className="bg-accent/10 text-accent border border-accent/40 px-4 py-2 rounded-lg text-sm hover:bg-accent/20"
        >
          Register Speaker
        </button>
      </div>

      <div className="bg-panel border border-panelBorder rounded-xl p-4 space-y-3">
        <h2 className="text-sm font-semibold text-slate-300">2. Test Verification</h2>
        <input type="file" accept="audio/*" onChange={(e) => setTestFile(e.target.files?.[0] ?? null)} />
        <button
          onClick={verify}
          className="bg-accent/10 text-accent border border-accent/40 px-4 py-2 rounded-lg text-sm hover:bg-accent/20"
        >
          Verify Against Reference
        </button>
        {result && (
          <div className="text-sm text-slate-300 mt-2 space-y-1">
            <div>Speaker Similarity: {(result.speaker_similarity * 100).toFixed(1)}%</div>
            <div>Identity Match: {result.identity_match ? "Yes" : "No"}</div>
            <div>Confidence: {(result.confidence * 100).toFixed(1)}%</div>
          </div>
        )}
      </div>

      {status && <p className="text-sm text-slate-500">{status}</p>}
    </div>
  );
}
