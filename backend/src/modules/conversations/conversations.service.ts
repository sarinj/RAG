import { randomUUID } from 'node:crypto';
import { Inject, Injectable, Logger, OnApplicationShutdown, OnModuleInit } from '@nestjs/common';
import { DataSource, EntityManager } from 'typeorm';
import { ServerConfig } from '../../config/configuration';
import { readIndexingArtifact } from '../../config/index-config';
import { DATA_SOURCE, SERVER_CONFIG } from '../../config/tokens';
import { decodeCursor, pageResult } from '../../common/pagination';
import { ApiError, badRequest, conflict, notFound, unavailable } from '../../common/errors/api-error';
import { InternalCitation, InternalChatResponse } from '../llm-client/contracts/chat.contract';
import { LlmClientService } from '../llm-client/llm-client.service';

type Scope = { mode: 'all' } | { mode: 'selected'; documentIds: string[] };
type SourceSnapshot = { documentId: string; generationId: string };
type ConversationRow = { id: string; title: string; scope_mode: 'all' | 'selected'; scope: Scope; created_at: Date | string; updated_at: Date | string };
type TurnRow = { id: string; conversation_id: string; request_id: string; question: string; status: string; sources: SourceSnapshot[]; expires_at: Date | string };
type CitationRow = {
  id: string; generation_id: string; document_id: string; document_name: string;
  page_start: number | null; page_end: number | null; heading: string | null;
  generation_status: string; active_generation_id: string | null; deleted_at: Date | null;
};
type PublicCitation = {
  label: string; documentId: string; generationId: string; chunkId: string;
  documentName: string; pageStart: number | null; pageEnd: number | null; heading: string | null;
};
type PublicMessage = {
  id: string; conversationId: string; requestId: string; role: string; status: string; content: string;
  answerStatus: string | null; citations: PublicCitation[]; error: { code: string; message: string } | null;
  createdAt: string; updatedAt: string;
};
type PublicTurn = { requestId: string; userMessage: PublicMessage; assistantMessage: PublicMessage };

@Injectable()
export class ConversationsService implements OnModuleInit, OnApplicationShutdown {
  private readonly logger = new Logger(ConversationsService.name);
  private recoveryTimer?: NodeJS.Timeout;
  private recovering = false;

  constructor(
    @Inject(DATA_SOURCE) private readonly dataSource: DataSource,
    @Inject(SERVER_CONFIG) private readonly config: ServerConfig,
    private readonly llm: LlmClientService,
  ) {}

  onModuleInit(): void {
    this.recoveryTimer = setInterval(() => void this.recoverExpiredTurns(), 15_000);
    this.recoveryTimer.unref();
    void this.recoverExpiredTurns();
  }

  onApplicationShutdown(): void {
    if (this.recoveryTimer) clearInterval(this.recoveryTimer);
  }

  async create(input: { title?: string; scope: unknown }): Promise<Record<string, unknown>> {
    const scope = parseScope(input.scope);
    if (input.title !== undefined && typeof input.title !== 'string') throw badRequest('Conversation title must be text.');
    const title = input.title === undefined ? 'New chat' : input.title.trim();
    if (!title || [...title].length > 120) throw badRequest('Conversation title must be between 1 and 120 characters.');
    const id = randomUUID();

    await this.dataSource.transaction(async (manager) => {
      if (scope.mode === 'selected') {
        const documents = await manager.query<{ id: string; status: string; deleted_at: Date | null }[]>(
          `SELECT id,status,deleted_at FROM documents WHERE id = ANY($1::uuid[]) ORDER BY id FOR UPDATE`,
          [scope.documentIds],
        );
        if (documents.length !== scope.documentIds.length) throw notFound('One or more selected documents were not found.');
        if (documents.some((document) => document.status !== 'ready' || document.deleted_at)) {
          throw conflict('DOCUMENT_NOT_READY', 'Every selected document must be ready.');
        }
      }
      await manager.query(
        `INSERT INTO conversations (id,title,scope_mode) VALUES ($1,$2,$3)`, [id, title, scope.mode],
      );
      if (scope.mode === 'selected') {
        for (const documentId of scope.documentIds) {
          await manager.query(
            `INSERT INTO conversation_documents (conversation_id,document_id) VALUES ($1,$2)`, [id, documentId],
          );
        }
      }
    });
    return this.get(id);
  }

  async list(limit: number, cursor?: string): Promise<{ items: Record<string, unknown>[]; nextCursor: string | null }> {
    const position = decodeCursor(cursor);
    const params: unknown[] = [];
    let predicate = '';
    if (position) {
      params.push(position.createdAt, position.id);
      predicate = `WHERE (c.created_at,c.id) < ($1::timestamptz,$2::uuid)`;
    }
    params.push(limit + 1);
    const rows = await this.dataSource.query<ConversationRow[]>(
      `${conversationSelect()} ${predicate} ORDER BY c.created_at DESC,c.id DESC LIMIT $${params.length}`,
      params,
    );
    const page = pageResult(rows.map(withCreatedAt), limit);
    return { items: page.rows.map(mapConversation), nextCursor: page.nextCursor };
  }

  async get(id: string): Promise<Record<string, unknown>> {
    const rows = await this.dataSource.query<ConversationRow[]>(`${conversationSelect()} WHERE c.id = $1 LIMIT 1`, [id]);
    if (!rows[0]) throw notFound('Conversation not found.');
    return mapConversation(rows[0]);
  }

  async messages(conversationId: string, limit: number, cursor?: string): Promise<{ items: PublicMessage[]; nextCursor: string | null }> {
    const exists = await this.dataSource.query<{ id: string }[]>(`SELECT id FROM conversations WHERE id = $1`, [conversationId]);
    if (!exists.length) throw notFound('Conversation not found.');
    const position = decodeCursor(cursor);
    const params: unknown[] = [conversationId];
    let predicate = 'm.conversation_id = $1';
    if (position) {
      params.push(position.createdAt, position.id);
      predicate += ` AND (m.created_at,m.id) > ($2::timestamptz,$3::uuid)`;
    }
    params.push(limit + 1);
    const rows = await this.dataSource.query<MessageRow[]>(
      `${messageSelect()} WHERE ${predicate} ORDER BY m.created_at ASC,m.id ASC LIMIT $${params.length}`,
      params,
    );
    const page = pageResult(rows.map(withCreatedAt), limit);
    return { items: page.rows.map(mapMessage), nextCursor: page.nextCursor };
  }

  async sendMessage(
    conversationId: string,
    requestId: string,
    rawContent: string,
    requestHeaderId: string,
  ): Promise<{ turn: PublicTurn; httpStatus: 200 | 202 }> {
    const content = rawContent.trim();
    if (!content || [...content].length > 8000) throw badRequest('Message content must be between 1 and 8000 characters.');

    const prior = await this.findTurn(conversationId, requestId);
    if (prior) {
      if (prior.question !== content) throw conflict('REQUEST_ID_CONFLICT', 'This request ID was already used with different content.');
      return { turn: await this.getTurn(prior.id), httpStatus: prior.status === 'pending' ? 202 : 200 };
    }

    const conversation = await this.getConversationRecord(conversationId);
    const created = await this.createPendingTurn(conversation, requestId, content);
    if (created.duplicate) {
      const duplicate = await this.findTurn(conversationId, requestId);
      if (!duplicate) throw unavailable('INTERNAL_ERROR', 'The chat turn could not be loaded.');
      if (duplicate.question !== content) throw conflict('REQUEST_ID_CONFLICT', 'This request ID was already used with different content.');
      return { turn: await this.getTurn(duplicate.id), httpStatus: duplicate.status === 'pending' ? 202 : 200 };
    }

    try {
      const response = await this.llm.chat({
        contractVersion: '1.1.0',
        requestId,
        conversationId,
        question: content,
        history: created.history,
        sources: created.sources,
        retrievalMode: this.indexingArtifact().retrievalMode,
      }, requestHeaderId);
      const citationRows = await this.validateCitations(response, created.sources);
      await this.completeTurn(created.turnId, response, citationRows);
      return { turn: await this.getTurn(created.turnId), httpStatus: 200 };
    } catch (error) {
      const failure = error instanceof ApiError
        ? error
        : unavailable('MODEL_UNAVAILABLE', 'The internal model service is unavailable.');
      await this.failTurn(created.turnId, failure.code, failure.message);
      throw failure;
    }
  }

  private async createPendingTurn(
    conversation: { id: string; scope_mode: 'all' | 'selected'; scope: Scope },
    requestId: string,
    question: string,
  ): Promise<{ turnId: string; sources: SourceSnapshot[]; history: { role: 'user' | 'assistant'; content: string }[]; duplicate: boolean }> {
    return this.dataSource.transaction(async (manager) => {
      const sourceRows = await this.lockAndResolveSources(manager, conversation.scope);
      const sources = sourceRows.map((row) => ({ documentId: row.id, generationId: row.active_generation_id as string }));
      await this.assertCompatibleIndex(sourceRows.map((row) => row.generation_id));

      await manager.query(`SELECT id FROM conversations WHERE id = $1 FOR UPDATE`, [conversation.id]);
      const existing = await manager.query<TurnRow[]>(
        `SELECT id,conversation_id,request_id,question,status,sources,expires_at
         FROM chat_turns WHERE conversation_id = $1 AND request_id = $2 LIMIT 1`, [conversation.id, requestId],
      );
      if (existing[0]) {
        if (existing[0].question !== question) throw conflict('REQUEST_ID_CONFLICT', 'This request ID was already used with different content.');
        return { turnId: existing[0].id, sources: existing[0].sources, history: [], duplicate: true };
      }
      const pending = await manager.query<{ id: string }[]>(
        `SELECT id FROM chat_turns WHERE conversation_id = $1 AND status = 'pending' LIMIT 1`, [conversation.id],
      );
      if (pending.length) throw conflict('TURN_IN_PROGRESS', 'Another answer is still being generated for this conversation.');

      const historyRows = await manager.query<{ role: 'user' | 'assistant'; content: string }[]>(
        `SELECT m.role,m.content
         FROM messages m JOIN chat_turns t ON t.id = m.turn_id
         WHERE m.conversation_id = $1 AND m.status = 'completed' AND t.status = 'completed'
         ORDER BY t.created_at DESC,t.id DESC,CASE m.role WHEN 'assistant' THEN 1 ELSE 0 END DESC LIMIT 20`, [conversation.id],
      );
      const history = historyRows.reverse();
      const turnId = randomUUID();
      const userMessageId = randomUUID();
      const assistantMessageId = randomUUID();
      const expiresAt = new Date(Date.now() + this.config.turnExpirySeconds * 1000);
      try {
        await manager.query(
          `INSERT INTO chat_turns (id,conversation_id,request_id,question,sources,status,expires_at)
           VALUES ($1,$2,$3,$4,$5::jsonb,'pending',$6)`,
          [turnId, conversation.id, requestId, question, JSON.stringify(sources), expiresAt],
        );
      } catch (error) {
        if (isUniqueViolation(error)) throw conflict('TURN_IN_PROGRESS', 'Another answer is still being generated for this conversation.');
        throw error;
      }
      await manager.query(
        `INSERT INTO messages (id,turn_id,conversation_id,role,status,content)
         VALUES ($1,$2,$3,'user','completed',$4),($5,$2,$3,'assistant','pending','')`,
        [userMessageId, turnId, conversation.id, question, assistantMessageId],
      );
      await manager.query(`UPDATE conversations SET updated_at = now() WHERE id = $1`, [conversation.id]);
      return { turnId, sources, history, duplicate: false };
    });
  }

  private async lockAndResolveSources(
    manager: EntityManager,
    scope: Scope,
  ): Promise<{ id: string; active_generation_id: string | null; generation_id: string; status: string; deleted_at: Date | null }[]> {
    let rows: { id: string; active_generation_id: string | null; status: string; deleted_at: Date | null }[];
    if (scope.mode === 'all') {
      rows = await manager.query(
        `SELECT id,active_generation_id,status,deleted_at FROM documents
         WHERE status = 'ready' AND deleted_at IS NULL AND active_generation_id IS NOT NULL
         ORDER BY id FOR UPDATE`,
      );
      if (!rows.length) throw conflict('NO_READY_DOCUMENTS', 'There are no ready documents to search.');
    } else {
      rows = await manager.query(
        `SELECT id,active_generation_id,status,deleted_at FROM documents
         WHERE id = ANY($1::uuid[]) ORDER BY id FOR UPDATE`, [scope.documentIds],
      );
      if (rows.length !== scope.documentIds.length || rows.some((row) => row.status !== 'ready' || row.deleted_at || !row.active_generation_id)) {
        throw conflict('DOCUMENT_NOT_READY', 'Every selected document must be ready before asking a question.');
      }
    }

    const generationIds = rows.map((row) => row.active_generation_id as string);
    const generations = await manager.query<{ id: string; document_id: string; status: string }[]>(
      `SELECT id,document_id,status FROM index_generations WHERE id = ANY($1::uuid[]) ORDER BY document_id`, [generationIds],
    );
    const generationByDocument = new Map(generations.map((generation) => [generation.document_id, generation]));
    return rows.map((row) => {
      const generation = generationByDocument.get(row.id);
      if (!generation || generation.status !== 'ready') throw conflict('DOCUMENT_NOT_READY', 'Every selected document must be ready before asking a question.');
      return { ...row, generation_id: generation.id };
    });
  }

  private async assertCompatibleIndex(generationIds: string[]): Promise<void> {
    const artifact = this.indexingArtifact();
    const generations = await this.dataSource.query<{ embedding_model: string; embedding_digest: string; embedding_dimension: number }[]>(
      `SELECT embedding_model,embedding_digest,embedding_dimension FROM index_generations WHERE id = ANY($1::uuid[])`, [generationIds],
    );
    const compatible = generations.every((generation) =>
      generation.embedding_model === artifact.embeddingModel
      && generation.embedding_digest === artifact.embeddingDigest
      && Number(generation.embedding_dimension) === artifact.embeddingDimension,
    );
    if (!compatible || generations.length !== generationIds.length) {
      throw conflict('INDEX_CONFIG_MISMATCH', 'The selected documents use an embedding configuration that is not active.');
    }
  }

  private indexingArtifact() {
    try {
      return readIndexingArtifact(this.config);
    } catch {
      throw unavailable('MODEL_UNAVAILABLE', 'The shared indexing configuration is missing or invalid.');
    }
  }

  private async validateCitations(response: InternalChatResponse, sources: SourceSnapshot[]): Promise<CitationRow[]> {
    const citations = response.citations;
    validateCitationMarkers(response.answerStatus, response.answer, citations.map((citation) => citation.label));
    const labels = new Set<string>();
    for (const citation of citations) {
      if (!/^S[1-9][0-9]*$/.test(citation.label) || labels.has(citation.label)) {
        throw new ApiError('INVALID_CITATIONS', 'The generated answer contains invalid source labels.', 422);
      }
      labels.add(citation.label);
    }
    if (!citations.length) return [];
    const sourceKeys = new Set(sources.map((source) => `${source.documentId}:${source.generationId}`));
    if (citations.some((citation) => !sourceKeys.has(`${citation.documentId}:${citation.generationId}`))) {
      throw new ApiError('INVALID_CITATIONS', 'A source reference is outside the selected documents.', 422);
    }
    const chunkIds = [...new Set(citations.map((citation) => citation.chunkId))];
    const rows = await this.dataSource.query<CitationRow[]>(
      `SELECT c.id,c.generation_id,g.document_id,d.name AS document_name,c.page_start,c.page_end,c.heading,
              g.status AS generation_status,d.active_generation_id,d.deleted_at
       FROM chunks c
       JOIN index_generations g ON g.id = c.generation_id
       JOIN documents d ON d.id = g.document_id
       WHERE c.id = ANY($1::uuid[])`, [chunkIds],
    );
    const rowById = new Map(rows.map((row) => [row.id, row]));
    const enriched: CitationRow[] = [];
    for (const citation of citations) {
      const row = rowById.get(citation.chunkId);
      if (!row || row.document_id !== citation.documentId || row.generation_id !== citation.generationId
        || row.deleted_at || row.generation_status !== 'ready') {
        throw new ApiError('INVALID_CITATIONS', 'A source reference did not resolve to a valid excerpt.', 422);
      }
      enriched.push(row);
    }
    return enriched;
  }

  private async completeTurn(turnId: string, response: InternalChatResponse, citationRows: CitationRow[]): Promise<void> {
    const turn = await this.dataSource.query<TurnRow[]>(
      `SELECT id,conversation_id,request_id,question,status,sources,expires_at FROM chat_turns WHERE id = $1`, [turnId],
    );
    if (!turn[0]) throw notFound('Chat turn not found.');
    const snapshot = turn[0].sources;
    let stale = false;
    let expired = false;
    await this.dataSource.transaction(async (manager) => {
      const sourceIds = snapshot.map((source) => source.documentId).sort();
      const documents = await manager.query<{ id: string; status: string; active_generation_id: string | null; deleted_at: Date | null }[]>(
        `SELECT id,status,active_generation_id,deleted_at FROM documents
         WHERE id = ANY($1::uuid[]) ORDER BY id FOR UPDATE`, [sourceIds],
      );
      const expected = new Map(snapshot.map((source) => [source.documentId, source.generationId]));
      stale = documents.length !== snapshot.length || documents.some((document) =>
        document.status !== 'ready' || document.deleted_at || document.active_generation_id !== expected.get(document.id),
      );
      const lockedTurn = await manager.query<{ status: string; expires_at: Date | string }[]>(
        `SELECT status,expires_at FROM chat_turns WHERE id = $1 FOR UPDATE`, [turnId],
      );
      if (!lockedTurn[0] || lockedTurn[0].status !== 'pending') return;
      expired = new Date(lockedTurn[0].expires_at).getTime() <= Date.now();
      if (stale || expired) {
        const errorCode = expired ? 'GENERATION_TIMEOUT' : 'SCOPE_CHANGED';
        const errorMessage = expired ? 'The answer generation timed out.' : 'The selected documents changed while the answer was being generated.';
        await manager.query(`UPDATE chat_turns SET status = 'failed',updated_at = now() WHERE id = $1`, [turnId]);
        await manager.query(
          `UPDATE messages SET status = 'failed',content = '',answer_status = NULL,error_code = $2,error_message = $3,updated_at = now()
           WHERE turn_id = $1 AND role = 'assistant'`, [turnId, errorCode, errorMessage],
        );
        return;
      }
      await manager.query(`UPDATE chat_turns SET status = 'completed',updated_at = now() WHERE id = $1`, [turnId]);
      await manager.query(
        `UPDATE messages SET status = 'completed',content = $2,answer_status = $3,error_code = NULL,error_message = NULL,updated_at = now()
         WHERE turn_id = $1 AND role = 'assistant'`, [turnId, response.answer, response.answerStatus],
      );
      const assistant = await manager.query<{ id: string }[]>(
        `SELECT id FROM messages WHERE turn_id = $1 AND role = 'assistant'`, [turnId],
      );
      for (let index = 0; index < response.citations.length; index += 1) {
        const citation: InternalCitation = response.citations[index];
        const source = citationRows[index];
        await manager.query(
          `INSERT INTO message_citations (message_id,label,document_id,generation_id,chunk_id,document_name,page_start,page_end,heading)
           VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)`,
          [assistant[0].id, citation.label, source.document_id, source.generation_id, source.id,
            source.document_name, source.page_start, source.page_end, source.heading],
        );
      }
    });
    if (stale) throw conflict('SCOPE_CHANGED', 'The selected documents changed while the answer was being generated.');
    if (expired) throw new ApiError('GENERATION_TIMEOUT', 'The answer generation timed out.', 504);
  }

  private async failTurn(turnId: string, code: string, message: string): Promise<void> {
    await this.dataSource.transaction(async (manager) => {
      const turn = await manager.query<{ status: string }[]>(
        `SELECT status FROM chat_turns WHERE id = $1 FOR UPDATE`, [turnId],
      );
      if (!turn[0] || turn[0].status !== 'pending') return;
      await manager.query(`UPDATE chat_turns SET status = 'failed',updated_at = now() WHERE id = $1`, [turnId]);
      await manager.query(
        `UPDATE messages SET status = 'failed',content = '',answer_status = NULL,error_code = $2,error_message = $3,updated_at = now()
         WHERE turn_id = $1 AND role = 'assistant' AND status = 'pending'`, [turnId, code, message],
      );
    });
  }

  private async findTurn(conversationId: string, requestId: string): Promise<TurnRow | undefined> {
    const rows = await this.dataSource.query<TurnRow[]>(
      `SELECT id,conversation_id,request_id,question,status,sources,expires_at
       FROM chat_turns WHERE conversation_id = $1 AND request_id = $2 LIMIT 1`, [conversationId, requestId],
    );
    return rows[0];
  }

  private async getConversationRecord(id: string): Promise<{ id: string; scope_mode: 'all' | 'selected'; scope: Scope }> {
    const rows = await this.dataSource.query<ConversationRow[]>(`${conversationSelect()} WHERE c.id = $1 LIMIT 1`, [id]);
    if (!rows[0]) throw notFound('Conversation not found.');
    return { id: rows[0].id, scope_mode: rows[0].scope_mode, scope: rows[0].scope };
  }

  private async getTurn(turnId: string): Promise<PublicTurn> {
    const rows = await this.dataSource.query<MessageRow[]>(
      `${messageSelect()} WHERE m.turn_id = $1 ORDER BY CASE m.role WHEN 'user' THEN 0 ELSE 1 END`, [turnId],
    );
    const userMessage = rows.find((row) => row.role === 'user');
    const assistantMessage = rows.find((row) => row.role === 'assistant');
    if (!userMessage || !assistantMessage) throw notFound('Chat turn not found.');
    return {
      requestId: userMessage.request_id,
      userMessage: mapMessage(userMessage),
      assistantMessage: mapMessage(assistantMessage),
    };
  }

  private async recoverExpiredTurns(): Promise<void> {
    if (this.recovering) return;
    this.recovering = true;
    try {
      await this.dataSource.transaction(async (manager) => {
        const expired = await manager.query<{ id: string }[]>(
          `SELECT id FROM chat_turns WHERE status = 'pending' AND expires_at <= now()
           ORDER BY expires_at FOR UPDATE SKIP LOCKED LIMIT 100`,
        );
        if (!expired.length) return;
        const ids = expired.map((turn) => turn.id);
        await manager.query(`UPDATE chat_turns SET status = 'failed',updated_at = now() WHERE id = ANY($1::uuid[])`, [ids]);
        await manager.query(
          `UPDATE messages SET status = 'failed',content = '',answer_status = NULL,
             error_code = 'GENERATION_TIMEOUT',error_message = 'The answer generation timed out.',updated_at = now()
           WHERE turn_id = ANY($1::uuid[]) AND role = 'assistant' AND status = 'pending'`, [ids],
        );
      });
    } catch (error) {
      this.logger.warn(`Expired turn recovery failed: ${error instanceof Error ? error.message : 'database error'}`);
    } finally {
      this.recovering = false;
    }
  }
}

type MessageRow = {
  id: string; conversation_id: string; request_id: string; role: string; status: string; content: string;
  answer_status: string | null; error_code: string | null; error_message: string | null;
  created_at: Date | string; updated_at: Date | string; citations: PublicCitation[];
};

function conversationSelect(): string {
  return `SELECT c.id,c.title,c.scope_mode,c.created_at,c.updated_at,
    CASE WHEN c.scope_mode = 'all' THEN jsonb_build_object('mode','all')
         ELSE jsonb_build_object('mode','selected','documentIds',
           COALESCE((SELECT jsonb_agg(cd.document_id::text ORDER BY cd.document_id)
                     FROM conversation_documents cd WHERE cd.conversation_id = c.id),'[]'::jsonb))
    END AS scope
    FROM conversations c`;
}

function messageSelect(): string {
  return `SELECT m.id,m.conversation_id,t.request_id,m.role,m.status,m.content,m.answer_status,
    m.error_code,m.error_message,m.created_at,m.updated_at,
    COALESCE((SELECT jsonb_agg(jsonb_build_object(
      'label',mc.label,'documentId',mc.document_id,'generationId',mc.generation_id,'chunkId',mc.chunk_id,
      'documentName',mc.document_name,'pageStart',mc.page_start,'pageEnd',mc.page_end,'heading',mc.heading
    ) ORDER BY mc.label) FROM message_citations mc WHERE mc.message_id = m.id),'[]'::jsonb) AS citations
    FROM messages m JOIN chat_turns t ON t.id = m.turn_id`;
}

function mapConversation(row: ConversationRow): Record<string, unknown> {
  return {
    id: row.id,
    title: row.title,
    scope: row.scope,
    createdAt: new Date(row.created_at).toISOString(),
    updatedAt: new Date(row.updated_at).toISOString(),
  };
}

function mapMessage(row: MessageRow): PublicMessage {
  return {
    id: row.id,
    conversationId: row.conversation_id,
    requestId: row.request_id,
    role: row.role,
    status: row.status,
    content: row.content,
    answerStatus: row.answer_status,
    citations: Array.isArray(row.citations) ? row.citations : [],
    error: row.error_code ? { code: row.error_code, message: row.error_message ?? 'The answer could not be generated.' } : null,
    createdAt: new Date(row.created_at).toISOString(),
    updatedAt: new Date(row.updated_at).toISOString(),
  };
}

export function parseScope(raw: unknown): Scope {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw badRequest('A conversation scope is required.');
  const value = raw as Record<string, unknown>;
  if (value.mode === 'all') {
    if (Object.keys(value).length !== 1) throw badRequest('The all-documents scope does not accept document IDs.');
    return { mode: 'all' };
  }
  if (value.mode !== 'selected' || Object.keys(value).sort().join(',') !== 'documentIds,mode') {
    throw badRequest('The conversation scope is invalid.');
  }
  if (!Array.isArray(value.documentIds) || value.documentIds.length < 1 || value.documentIds.length > 100) {
    throw badRequest('Selected scope requires between 1 and 100 document IDs.');
  }
  const ids = value.documentIds;
  if (ids.some((id) => typeof id !== 'string' || !UUID.test(id))) throw badRequest('Every selected document ID must be a UUID.');
  const normalized = (ids as string[]).map((id) => id.toLowerCase());
  if (new Set(normalized).size !== normalized.length) throw badRequest('Selected document IDs must be unique.');
  return { mode: 'selected', documentIds: normalized.sort() };
}

function withCreatedAt<T extends { created_at: Date | string; id: string }>(row: T): T & { createdAt: Date | string } {
  return { ...row, createdAt: row.created_at };
}

function isUniqueViolation(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && (error as { code?: unknown }).code === '23505';
}

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export function validateCitationMarkers(
  answerStatus: 'answered' | 'insufficient_evidence' | 'direct_answer',
  answer: string,
  citationLabels: string[],
): void {
  const markers = new Set(Array.from(answer.matchAll(/\[(S[^\]\s]*)\]/g), (match) => match[1]));
  const labels = new Set(citationLabels);
  if (markers.size !== labels.size || [...markers].some((label) => !labels.has(label))) {
    throw new ApiError('INVALID_CITATIONS', 'The generated answer contains invalid source references.', 422);
  }
  if (!answer.trim()) {
    throw new ApiError('INVALID_CITATIONS', 'The generated answer is empty.', 422);
  }
  if (answerStatus === 'answered' && labels.size === 0) {
    throw new ApiError('INVALID_CITATIONS', 'An answer needs at least one valid citation.', 422);
  }
  if (answerStatus !== 'answered' && labels.size > 0) {
    throw new ApiError('INVALID_CITATIONS', 'A non-evidence answer cannot include citations.', 422);
  }
}
