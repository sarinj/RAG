import { OpenAPIObject } from '@nestjs/swagger';

export function addContractErrors(document: OpenAPIObject): OpenAPIObject {
  const failures: Record<string, Record<string, Record<string, string[]>>> = {
    '/api/documents': {
      get: { '400': ['VALIDATION_ERROR'], '500': ['INTERNAL_ERROR'] },
      post: {
        '400': ['VALIDATION_ERROR'], '409': ['DUPLICATE_DOCUMENT'],
        '413': ['FILE_TOO_LARGE', 'CORPUS_LIMIT_EXCEEDED'], '415': ['UNSUPPORTED_FILE_TYPE'],
        '503': ['MODEL_UNAVAILABLE'], '500': ['INTERNAL_ERROR'],
      },
    },
    '/api/documents/{id}': {
      get: { '400': ['VALIDATION_ERROR'], '404': ['NOT_FOUND'], '500': ['INTERNAL_ERROR'] },
      delete: { '400': ['VALIDATION_ERROR'], '404': ['NOT_FOUND'], '500': ['INTERNAL_ERROR'] },
    },
    '/api/documents/{id}/retry': {
      post: { '400': ['VALIDATION_ERROR'], '404': ['NOT_FOUND'], '409': ['INVALID_STATE'], '500': ['INTERNAL_ERROR'] },
    },
    '/api/documents/{id}/reindex': {
      post: {
        '400': ['VALIDATION_ERROR'], '404': ['NOT_FOUND'], '409': ['INVALID_STATE', 'DOCUMENT_NOT_READY'],
        '503': ['MODEL_UNAVAILABLE'], '500': ['INTERNAL_ERROR'],
      },
    },
    '/api/documents/{id}/chunks/{chunkId}': {
      get: { '400': ['VALIDATION_ERROR'], '404': ['NOT_FOUND'], '410': ['SOURCE_DELETED'], '500': ['INTERNAL_ERROR'] },
    },
    '/api/conversations': {
      get: { '400': ['VALIDATION_ERROR'], '500': ['INTERNAL_ERROR'] },
      post: { '400': ['VALIDATION_ERROR'], '404': ['NOT_FOUND'], '409': ['DOCUMENT_NOT_READY'], '500': ['INTERNAL_ERROR'] },
    },
    '/api/conversations/{id}': {
      get: { '400': ['VALIDATION_ERROR'], '404': ['NOT_FOUND'], '500': ['INTERNAL_ERROR'] },
    },
    '/api/conversations/{id}/messages': {
      get: { '400': ['VALIDATION_ERROR'], '404': ['NOT_FOUND'], '500': ['INTERNAL_ERROR'] },
      post: {
        '400': ['VALIDATION_ERROR'], '404': ['NOT_FOUND'],
        '409': ['NO_READY_DOCUMENTS', 'DOCUMENT_NOT_READY', 'REQUEST_ID_CONFLICT', 'TURN_IN_PROGRESS', 'SCOPE_CHANGED', 'INDEX_CONFIG_MISMATCH'],
        '422': ['CONTEXT_LIMIT_EXCEEDED', 'INVALID_CITATIONS'], '503': ['MODEL_UNAVAILABLE'],
        '504': ['GENERATION_TIMEOUT'], '500': ['INTERNAL_ERROR'],
      },
    },
    '/health/ready': { get: { '503': ['MODEL_UNAVAILABLE'] } },
  };
  for (const [pathName, path] of Object.entries(document.paths)) {
    for (const [method, operation] of Object.entries(path ?? {})) {
      if (!operation || typeof operation !== 'object' || !('responses' in operation)) continue;
      const routeFailures = failures[pathName]?.[method];
      if (!routeFailures) continue;
      const responses = operation.responses as Record<string, unknown>;
      for (const [status, codes] of Object.entries(routeFailures)) {
        const current = responses[status] && typeof responses[status] === 'object'
          ? responses[status] as Record<string, unknown>
          : {};
        const currentContent = current.content && typeof current.content === 'object'
          ? current.content as Record<string, unknown>
          : {};
        const currentJson = currentContent['application/json'] && typeof currentContent['application/json'] === 'object'
          ? currentContent['application/json'] as Record<string, unknown>
          : {};
        responses[status] = {
          ...current,
          description: current.description ?? codes.join(', '),
          content: {
            ...currentContent,
            'application/json': {
              ...currentJson,
              schema: { $ref: '#/components/schemas/ErrorEnvelope' },
              examples: Object.fromEntries(codes.map((code) => [code, {
                value: { code, message: 'Request failed.', requestId: '00000000-0000-4000-8000-000000000000' },
              }])),
            },
          },
        };
      }
    }
  }
  document.components ??= {};
  document.components.schemas ??= {};
  document.components.schemas.ErrorEnvelope = {
    type: 'object',
    required: ['code', 'message', 'requestId'],
    properties: {
      code: { type: 'string' },
      message: { type: 'string' },
      requestId: { type: 'string', format: 'uuid' },
    },
    additionalProperties: false,
  };
  for (const name of ['CreateConversationDto', 'CreateMessageDto', 'EmptyDto']) {
    const schema = document.components.schemas[name];
    if (schema && typeof schema === 'object' && 'type' in schema && schema.type === 'object') {
      (schema as Record<string, unknown>).additionalProperties = false;
    }
  }
  return document;
}
