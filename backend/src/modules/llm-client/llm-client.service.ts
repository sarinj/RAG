import { Inject, Injectable } from '@nestjs/common';
import { ServerConfig } from '../../config/configuration';
import { SERVER_CONFIG } from '../../config/tokens';
import { ApiError, unavailable } from '../../common/errors/api-error';
import { readServerEvents } from '../../common/sse';
import { InternalChatRequest, InternalChatResponse, InternalChatResponseSchema } from './contracts/chat.contract';
import { z } from 'zod';
import { InternalStreamEventSchema, LiveAnswerEvent } from './contracts/stream.contract';

const InternalReadyResponseSchema = z.object({ status: z.literal('ready') }).strict();

@Injectable()
export class LlmClientService {
  constructor(@Inject(SERVER_CONFIG) private readonly config: ServerConfig) {}

  async chat(request: InternalChatRequest, requestHeaderId: string): Promise<InternalChatResponse> {
    if (!this.config.internalApiToken) {
      throw unavailable('MODEL_UNAVAILABLE', 'The internal model service is not configured.');
    }

    let response: Response;
    try {
      response = await fetch(`${this.config.llmBaseUrl}/internal/chat`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${this.config.internalApiToken}`,
          'Content-Type': 'application/json',
          'X-Request-Id': requestHeaderId,
        },
        body: JSON.stringify(request),
        signal: AbortSignal.timeout(this.config.internalTimeoutMs),
      });
    } catch (error) {
      if (error instanceof Error && (error.name === 'TimeoutError' || error.name === 'AbortError')) {
        throw new ApiError('GENERATION_TIMEOUT', 'The answer generation timed out.', 504);
      }
      throw unavailable('MODEL_UNAVAILABLE', 'The internal model service is unavailable.');
    }

    const body: unknown = await response.json().catch(() => undefined);
    if (!response.ok) throw mapInternalError(response.status, body);
    const parsed = InternalChatResponseSchema.safeParse(body);
    if (!parsed.success || parsed.data.requestId !== request.requestId) {
      throw unavailable('MODEL_UNAVAILABLE', 'The internal model service returned an invalid response.');
    }
    return parsed.data;
  }

  async ready(): Promise<boolean> {
    if (!this.config.internalApiToken) return false;
    try {
      const response = await fetch(`${this.config.llmBaseUrl}/health/ready`, {
        signal: AbortSignal.timeout(2_000),
      });
      if (!response.ok) return false;
      const body: unknown = await response.json().catch(() => undefined);
      return InternalReadyResponseSchema.safeParse(body).success;
    } catch {
      return false;
    }
  }

  async chatStream(
    request: InternalChatRequest,
    requestHeaderId: string,
    onEvent: (event: LiveAnswerEvent) => Promise<void>,
  ): Promise<InternalChatResponse> {
    if (!this.config.internalApiToken) {
      throw unavailable('MODEL_UNAVAILABLE', 'The internal model service is not configured.');
    }
    try {
      const response = await fetch(`${this.config.llmBaseUrl}/internal/chat/stream`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${this.config.internalApiToken}`,
          'Content-Type': 'application/json', Accept: 'text/event-stream',
          'X-Request-Id': requestHeaderId,
        },
        body: JSON.stringify(request),
        signal: AbortSignal.timeout(this.config.internalTimeoutMs),
      });
      if (!response.ok) throw mapInternalError(response.status, await response.json().catch(() => undefined));
      if (!response.body || !response.headers.get('content-type')?.includes('text/event-stream')) {
        await response.body?.cancel().catch(() => undefined);
        throw unavailable('MODEL_UNAVAILABLE', 'The answer stream is unavailable.');
      }
      for await (const raw of readServerEvents(response.body)) {
        const parsed = InternalStreamEventSchema.safeParse(raw);
        if (!parsed.success || parsed.data.requestId !== request.requestId) {
          throw unavailable('MODEL_UNAVAILABLE', 'The model service returned an invalid stream event.');
        }
        const event = parsed.data;
        if (event.type === 'error') throw mapInternalError(event.status, event);
        if (event.type === 'result') {
          if (event.response.requestId !== request.requestId) {
            throw unavailable('MODEL_UNAVAILABLE', 'The model service returned a mismatched request ID.');
          }
          return event.response;
        }
        await onEvent(event);
      }
      throw unavailable('MODEL_UNAVAILABLE', 'The answer stream ended before completion.');
    } catch (error) {
      if (error instanceof ApiError) throw error;
      if (error instanceof Error && (error.name === 'TimeoutError' || error.name === 'AbortError')) {
        throw new ApiError('GENERATION_TIMEOUT', 'The answer generation timed out.', 504);
      }
      throw unavailable('MODEL_UNAVAILABLE', 'The answer stream could not be completed.');
    }
  }
}

function mapInternalError(status: number, body: unknown): ApiError {
  const candidate = typeof body === 'object' && body !== null && 'code' in body
    ? String((body as { code: unknown }).code)
    : '';
  switch (candidate) {
    case 'SCOPE_CHANGED':
      return new ApiError('SCOPE_CHANGED', 'The selected documents changed while the answer was being generated.', 409);
    case 'INDEX_CONFIG_MISMATCH':
      return new ApiError('INDEX_CONFIG_MISMATCH', 'The selected documents use incompatible index configurations.', 409);
    case 'CONTEXT_LIMIT_EXCEEDED':
      return new ApiError('CONTEXT_LIMIT_EXCEEDED', 'The question and selected evidence exceed the model context limit.', 422);
    case 'INVALID_CITATIONS':
      return new ApiError('INVALID_CITATIONS', 'The generated answer did not contain valid source citations.', 422);
    case 'GENERATION_TIMEOUT':
      return new ApiError('GENERATION_TIMEOUT', 'The answer generation timed out.', 504);
    case 'MODEL_UNAVAILABLE':
      return unavailable();
    default:
      return status === 504
        ? new ApiError('GENERATION_TIMEOUT', 'The answer generation timed out.', 504)
        : unavailable('MODEL_UNAVAILABLE', 'The internal model service is unavailable.');
  }
}
