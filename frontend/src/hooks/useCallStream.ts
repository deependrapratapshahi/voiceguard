import { useCallback, useEffect, useRef, useState } from "react";
import type { StreamResult, WSMessage } from "../types";

export function useCallStream(callId: string | null) {
  const [results, setResults] = useState<StreamResult[]>([]);
  const [connected, setConnected] = useState(false);
  const [speakerReferenceLoaded, setSpeakerReferenceLoaded] = useState(false);
  const [lastError, setLastError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!callId) return;

    setResults([]);
    setSpeakerReferenceLoaded(false);
    setLastError(null);

    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${window.location.host}/ws/calls/${callId}/audio`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (event) => {
      try {
        const data: WSMessage = JSON.parse(event.data);
        if (data.type === "connected") {
          setSpeakerReferenceLoaded(data.speaker_reference_loaded);
        } else if (data.type === "result") {
          setResults((prev) => [...prev.slice(-199), data]);
        } else if (data.type === "error") {
          setLastError(data.error);
        }
        // "context_updated" acks are informational only, no state change needed.
      } catch {
        // ignore malformed frames
      }
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [callId]);

  const sendAudioChunk = useCallback((chunk: ArrayBuffer) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(chunk);
    }
  }, []);

  const sendContextUpdate = useCallback((context: Record<string, unknown>) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(context));
    }
  }, []);

  return {
    results,
    connected,
    speakerReferenceLoaded,
    lastError,
    sendAudioChunk,
    sendContextUpdate,
  };
}
