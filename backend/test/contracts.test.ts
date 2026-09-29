import 'reflect-metadata';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import test from 'node:test';
import { OpenAPIObject } from '@nestjs/swagger';
import { decodeCursor, encodeCursor, pageResult } from '../src/common/pagination';
import { CHUNK_CAPACITY_LIMIT } from '../src/config/configuration';
import { IndexingArtifactSchema } from '../src/config/index-config';
import { InternalChatResponseSchema } from '../src/modules/llm-client/contracts/chat.contract';
import { LlmClientService } from '../src/modules/llm-client/llm-client.service';
import { ServerConfig } from '../src/config/configuration';
import { parseScope, validateCitationMarkers } from '../src/modules/conversations/conversations.service';
import { addContractErrors } from '../src/openapi/contract-errors';

const uuidA = '11111111-1111-4111-8111-111111111111';
const uuidB = '22222222-2222-4222-8222-222222222222';

test('pagination cursors round-trip and reject malformed values', () => {
  const position = { createdAt: '2026-09-29T03:00:00.000Z', id: uuidA };
  assert.deepEqual(decodeCursor(encodeCursor(position)), position);
  assert.throws(() => decodeCursor('not-a-cursor'), (error: unknown) => hasErrorCode(error, 'VALIDATION_ERROR'));
  const malformedUuid = encodeCursor({ ...position, id: 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa' });
  assert.throws(() => decodeCursor(malformedUuid), (error: unknown) => hasErrorCode(error, 'VALIDATION_ERROR'));

  const page = pageResult([
    { id: uuidA, createdAt: position.createdAt },
    { id: uuidB, createdAt: position.createdAt },
  ], 1);
  assert.equal(page.rows.length, 1);
  assert.ok(page.nextCursor);
});

test('conversation scopes are strict, unique, and stable', () => {
  assert.deepEqual(parseScope({ mode: 'all' }), { mode: 'all' });
  assert.deepEqual(parseScope({ mode: 'selected', documentIds: [uuidB, uuidA] }), {
    mode: 'selected', documentIds: [uuidA, uuidB],
  });
  assert.throws(() => parseScope({ mode: 'all', documentIds: [] }), (error: unknown) => hasErrorCode(error, 'VALIDATION_ERROR'));
  assert.throws(() => parseScope({ mode: 'selected', documentIds: [uuidA, uuidA] }), (error: unknown) => hasErrorCode(error, 'VALIDATION_ERROR'));
});

test('citation markers must exactly match source labels and answer status', () => {
  assert.doesNotThrow(() => validateCitationMarkers('answered', 'A grounded answer [S1].', ['S1']));
  assert.doesNotThrow(() => validateCitationMarkers('insufficient_evidence', 'I could not find that in the sources.', []));
  assert.throws(() => validateCitationMarkers('answered', 'An answer without a marker.', ['S1']), (error: unknown) => hasErrorCode(error, 'INVALID_CITATIONS'));
  assert.throws(() => validateCitationMarkers('answered', 'An answer [S2].', ['S1']), (error: unknown) => hasErrorCode(error, 'INVALID_CITATIONS'));
  assert.throws(() => validateCitationMarkers('insufficient_evidence', 'No evidence [S1].', ['S1']), (error: unknown) => hasErrorCode(error, 'INVALID_CITATIONS'));
});

test('index artifact requires a probed dimension and coherent chunk settings', () => {
  const artifact = {
    embeddingModel: 'schema-test-model',
    embeddingDigest: 'sha256:schema-test',
    embeddingDimension: 1, // Schema-only placeholder; deployment dimensions come from the reviewed model probe.
    chunkTargetTokens: 500,
    chunkOverlapTokens: 75,
    chunkerVersion: 'chunker-v1',
    tokenizerVersion: 'tokenizer-v1',
    contextPromptVersion: null,
    contextModel: null,
    contextModelDigest: null,
    contextWindowTokens: 8192,
    maxOutputTokens: 512,
  };
  assert.equal(IndexingArtifactSchema.safeParse(artifact).success, true);
  assert.equal(IndexingArtifactSchema.safeParse({ ...artifact, embeddingDimension: undefined }).success, false);
  assert.equal(IndexingArtifactSchema.safeParse({ ...artifact, chunkOverlapTokens: 500 }).success, false);
});

test('chunk capacity matches the fixed v1 contract limit', () => {
  assert.equal(CHUNK_CAPACITY_LIMIT, 5000);
});

test('internal responses enforce the v1 citation and answer shape', () => {
  const response = {
    contractVersion: '1.0.0',
    requestId: uuidA,
    answer: 'Text [S1].',
    answerStatus: 'answered',
    citations: [{
      label: 'S1', documentId: uuidB, generationId: uuidA, chunkId: uuidB,
      documentName: 'notes.md', pageStart: null, pageEnd: null, heading: 'Intro',
    }],
  };
  assert.equal(InternalChatResponseSchema.safeParse(response).success, true);
  assert.equal(InternalChatResponseSchema.safeParse({ ...response, extra: true }).success, false);
  assert.equal(InternalChatResponseSchema.safeParse({ ...response, contractVersion: '2.0.0' }).success, false);
  const citation = response.citations[0];
  assert.equal(InternalChatResponseSchema.safeParse({ ...response, citations: [{ ...citation, label: 'source' }] }).success, false);
  assert.equal(InternalChatResponseSchema.safeParse({ ...response, citations: [{ ...citation, documentName: '' }] }).success, false);
  assert.equal(InternalChatResponseSchema.safeParse({ ...response, citations: [{ ...citation, pageStart: 0 }] }).success, false);
  assert.equal(InternalChatResponseSchema.safeParse({ ...response, citations: [{ ...citation, pageStart: 4, pageEnd: 3 }] }).success, false);
});

test('internal readiness requires a valid ready response body', async () => {
  const client = new LlmClientService({
    llmBaseUrl: 'http://llm-backend.test',
    internalApiToken: 'test-token',
  } as ServerConfig);
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => new Response(JSON.stringify({ status: 'ok' }), { status: 200 });
    assert.equal(await client.ready(), false);
    globalThis.fetch = async () => new Response(JSON.stringify({ status: 'ready', extra: true }), { status: 200 });
    assert.equal(await client.ready(), false);
    globalThis.fetch = async () => new Response(JSON.stringify({ status: 'ready' }), { status: 200 });
    assert.equal(await client.ready(), true);
    globalThis.fetch = async () => new Response(JSON.stringify({ status: 'ready' }), { status: 503 });
    assert.equal(await client.ready(), false);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('OpenAPI error envelopes include route-specific failure codes', () => {
  const document: OpenAPIObject = {
    openapi: '3.0.0',
    info: { title: 'test', version: '1.0.0' },
    paths: {
      '/api/documents': { post: { responses: { '202': { description: 'accepted' }, '409': { description: 'conflict' } } } },
      '/health/live': { get: { responses: { '200': { description: 'ok' } } } },
    },
  };
  addContractErrors(document);
  const responses = document.paths['/api/documents']?.post?.responses;
  const conflictResponse = responses?.['409'];
  const internalResponse = responses?.['500'];
  assert.ok(conflictResponse && !('$ref' in conflictResponse));
  assert.ok(internalResponse && !('$ref' in internalResponse));
  assert.deepEqual(conflictResponse.content?.['application/json']?.examples?.DUPLICATE_DOCUMENT, {
    value: {
      code: 'DUPLICATE_DOCUMENT',
      message: 'Request failed.',
      requestId: '00000000-0000-4000-8000-000000000000',
    },
  });
  const errorSchema = internalResponse.content?.['application/json']?.schema;
  assert.ok(errorSchema && '$ref' in errorSchema);
  assert.equal(errorSchema.$ref, '#/components/schemas/ErrorEnvelope');
  assert.equal(document.paths['/health/live']?.get?.responses?.['409'], undefined);
});

test('exported OpenAPI keeps liveness narrow and health status required', async () => {
  const document = JSON.parse(await readFile(resolve('openapi.json'), 'utf8')) as OpenAPIObject;
  const liveResponses = document.paths['/health/live']?.get?.responses;
  assert.deepEqual(Object.keys(liveResponses ?? {}), ['200']);
  const liveResponse = liveResponses?.['200'];
  assert.ok(liveResponse && !('$ref' in liveResponse));
  const liveSchema = liveResponse.content?.['application/json']?.schema;
  assert.ok(liveSchema && 'required' in liveSchema);
  assert.ok(liveSchema.required?.includes('status'));

  const readyResponse = document.paths['/health/ready']?.get?.responses?.['200'];
  assert.ok(readyResponse && !('$ref' in readyResponse));
  const readySchema = readyResponse.content?.['application/json']?.schema;
  assert.ok(readySchema && 'required' in readySchema);
  assert.ok(readySchema.required?.includes('status'));
});

function hasErrorCode(error: unknown, code: string): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === code;
}
