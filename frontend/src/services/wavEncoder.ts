/**
 * Browser-side helper for the Live Call Monitoring demo: decodes an
 * uploaded audio file and slices it into small, self-contained WAV
 * chunks that can be streamed one-by-one over the /ws/calls/{id}/audio
 * WebSocket -- simulating a live call without any telephony
 * integration. Intended for authorized demo/test audio only.
 */

/** Encodes a mono Float32Array of PCM samples as a 16-bit WAV file. */
export function encodeWavPCM16(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample; // mono
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  function writeString(offset: number, str: string) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true); // byte rate
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true); // bits per sample
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }

  return buffer;
}

/**
 * Decodes an uploaded audio File and slices it into ~chunkSeconds-long
 * mono WAV chunks (as ArrayBuffers) ready to send over the WebSocket.
 * Multi-channel audio is downmixed to mono by averaging channels.
 */
export async function fileToWavChunks(file: File, chunkSeconds = 2.0): Promise<ArrayBuffer[]> {
  const arrayBuffer = await file.arrayBuffer();
  const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
  const audioContext = new AudioContextClass();

  let audioBuffer: AudioBuffer;
  try {
    audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
  } finally {
    await audioContext.close();
  }

  const sampleRate = audioBuffer.sampleRate;
  const numChannels = audioBuffer.numberOfChannels;
  const totalSamples = audioBuffer.length;

  // Downmix to mono.
  const mono = new Float32Array(totalSamples);
  for (let ch = 0; ch < numChannels; ch++) {
    const channelData = audioBuffer.getChannelData(ch);
    for (let i = 0; i < totalSamples; i++) {
      mono[i] += channelData[i] / numChannels;
    }
  }

  const chunkSamples = Math.floor(chunkSeconds * sampleRate);
  const chunks: ArrayBuffer[] = [];

  for (let start = 0; start < totalSamples; start += chunkSamples) {
    const end = Math.min(start + chunkSamples, totalSamples);
    // Skip a trailing fragment shorter than half a second -- too short
    // to be meaningfully analyzed.
    if (end - start < sampleRate * 0.5) break;
    const slice = mono.slice(start, end);
    chunks.push(encodeWavPCM16(slice, sampleRate));
  }

  return chunks;
}
