/** Decode bounded UTF-8 SSE frames across arbitrary network chunk boundaries. */
export async function* readServerEvents(body: ReadableStream<Uint8Array>): AsyncGenerator<unknown> {
  const reader = body.getReader();
  const decoder = new TextDecoder('utf-8', { fatal: true });
  let buffer = '';
  let data: string[] = [];
  let eventSize = 0;
  const limit = 262_144;
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      while (true) {
        const match = /[\r\n]/.exec(buffer);
        if (!match) break;
        const index = match.index;
        if (!done && buffer[index] === '\r' && index === buffer.length - 1) break;
        const line = buffer.slice(0, index);
        const delimiter = buffer[index] === '\r' && buffer[index + 1] === '\n' ? 2 : 1;
        buffer = buffer.slice(index + delimiter);
        if (line === '') {
          if (data.length) yield JSON.parse(data.join('\n')) as unknown;
          data = [];
          eventSize = 0;
        } else if (line.startsWith('data:')) {
          const text = line.slice(5).replace(/^ /, '');
          eventSize += text.length;
          if (eventSize > limit) throw new Error('Stream event exceeds its limit.');
          data.push(text);
        } else if (line.length > limit) {
          throw new Error('Stream line exceeds its limit.');
        }
      }
      if (buffer.length + eventSize > limit) throw new Error('Stream event exceeds its limit.');
      if (done) {
        if (buffer.trim() || data.length) throw new Error('Stream ended inside an event.');
        return;
      }
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
