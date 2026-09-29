import { createHash, randomUUID } from 'node:crypto';
import { TextDecoder } from 'node:util';
import { extname } from 'node:path';
import { Inject, Injectable, Logger, OnApplicationShutdown, OnModuleInit } from '@nestjs/common';
import { DataSource } from 'typeorm';
import { ServerConfig } from '../../config/configuration';
import { IndexingArtifact, readIndexingArtifact } from '../../config/index-config';
import { DATA_SOURCE, SERVER_CONFIG } from '../../config/tokens';
import { decodeCursor, pageResult } from '../../common/pagination';
import { ApiError, badRequest, conflict, notFound, unavailable } from '../../common/errors/api-error';
import { LocalFileStorage } from '../storage/storage.service';

type JobRow = {
  id: string;
  status: string;
  stage: string;
  processed_chunks: number;
  total_chunks: number | null;
  error_code: string | null;
  error_message: string | null;
};

type DocumentRow = {
  id: string;
  name: string;
  mime_type: string;
  size_bytes: string | number;
  status: string;
  active_generation_id: string | null;
  latest_job: JobRow | null;
  created_at: Date | string;
  updated_at: Date | string;
  storage_key?: string;
  deleted_at?: Date | string | null;
  cleanup_completed_at?: Date | string | null;
};

type PublicDocument = {
  id: string;
  name: string;
  mimeType: string;
  sizeBytes: number;
  status: string;
  activeGenerationId: string | null;
  latestJob: null | {
    id: string;
    status: string;
    stage: string;
    processedChunks: number;
    totalChunks: number | null;
    error: null | { code: string; message: string };
  };
  createdAt: string;
  updatedAt: string;
};

type UploadMetadata = { displayName: string; extension: '.pdf' | '.txt' | '.md'; mimeType: string };

@Injectable()
export class DocumentsService {
  constructor(
    @Inject(DATA_SOURCE) private readonly dataSource: DataSource,
    @Inject(SERVER_CONFIG) private readonly config: ServerConfig,
    private readonly storage: LocalFileStorage,
  ) {}

  async upload(file: Express.Multer.File): Promise<PublicDocument> {
    if (!file?.buffer) throw badRequest('Exactly one file is required.');
    if (file.size > this.config.maxUploadBytes) {
      throw new ApiError('FILE_TOO_LARGE', 'The uploaded file exceeds the configured size limit.', 413);
    }
    const upload = this.inspectUpload(file);
    const contentHash = createHash('sha256').update(file.buffer).digest('hex');
    const artifact = this.indexingArtifact();
    const stored = await this.storage.store(file.buffer, upload.extension);
    const documentId = randomUUID();

    try {
      await this.dataSource.transaction(async (manager) => {
        await manager.query('SELECT pg_advisory_xact_lock(71002)');
        const [{ count }] = await manager.query<{ count: string }[]>(
          `SELECT count(*)::text AS count FROM documents WHERE deleted_at IS NULL`,
        );
        if (Number(count) >= this.config.maxFiles) {
          throw new ApiError('CORPUS_LIMIT_EXCEEDED', 'The document limit has been reached.', 413);
        }
        const duplicate = await manager.query<{ id: string }[]>(
          `SELECT id FROM documents WHERE content_hash = $1 AND deleted_at IS NULL LIMIT 1`,
          [contentHash],
        );
        if (duplicate.length) throw conflict('DUPLICATE_DOCUMENT', 'A document with the same content already exists.');

        await manager.query(
          `INSERT INTO documents (id,name,mime_type,size_bytes,content_hash,storage_key,status)
           VALUES ($1,$2,$3,$4,$5,$6,'queued')`,
          [documentId, upload.displayName, upload.mimeType, file.size, contentHash, stored.storageKey],
        );
        await this.insertGenerationAndJob(manager, {
          documentId,
          generationId: randomUUID(),
          jobId: randomUUID(),
          artifact,
          mode: this.config.indexMode,
        });
      });
    } catch (error) {
      await this.storage.remove(stored.storageKey).catch(() => undefined);
      if (isUniqueViolation(error)) {
        throw conflict('DUPLICATE_DOCUMENT', 'A document with the same content already exists.');
      }
      throw error;
    }
    return this.get(documentId);
  }

  async list(limit: number, cursor?: string): Promise<{ items: PublicDocument[]; nextCursor: string | null }> {
    const position = decodeCursor(cursor);
    const params: unknown[] = [];
    let predicate = 'd.deleted_at IS NULL';
    if (position) {
      params.push(position.createdAt, position.id);
      predicate += ` AND (d.created_at,d.id) < ($${params.length - 1}::timestamptz,$${params.length}::uuid)`;
    }
    params.push(limit + 1);
    const rows = await this.dataSource.query<DocumentRow[]>(
      `${documentSelect()}
       WHERE ${predicate}
       ORDER BY d.created_at DESC,d.id DESC LIMIT $${params.length}`,
      params,
    );
    const page = pageResult(rows.map((row) => ({ ...row, createdAt: row.created_at })), limit);
    return { items: page.rows.map(mapDocument), nextCursor: page.nextCursor };
  }

  async get(id: string): Promise<PublicDocument> {
    const rows = await this.dataSource.query<DocumentRow[]>(`${documentSelect()} WHERE d.id = $1 LIMIT 1`, [id]);
    if (!rows[0]) throw notFound('Document not found.');
    return mapDocument(rows[0]);
  }

  async retry(id: string): Promise<PublicDocument> {
    await this.dataSource.transaction(async (manager) => {
      const documents = await manager.query<{ id: string; status: string; deleted_at: Date | null }[]>(
        `SELECT id,status,deleted_at FROM documents WHERE id = $1 FOR UPDATE`, [id],
      );
      const document = documents[0];
      if (!document) throw notFound('Document not found.');
      if (document.deleted_at || document.status === 'deleted') throw conflict('INVALID_STATE', 'A deleted document cannot be retried.');

      const active = await manager.query<{ id: string }[]>(
        `SELECT id FROM ingestion_jobs WHERE document_id = $1 AND status IN ('queued','processing') LIMIT 1`, [id],
      );
      if (active[0]) return;
      if (document.status !== 'failed' && document.status !== 'ready') {
        throw conflict('INVALID_STATE', 'Only failed ingestion work can be retried.');
      }

      const failed = await manager.query<{ latest_status: string; mode: string; embedding_model: string; embedding_digest: string; embedding_dimension: number; config: Record<string, unknown> }[]>(
        `SELECT j.status AS latest_status,g.mode,g.embedding_model,g.embedding_digest,g.embedding_dimension,g.config
         FROM index_generations g JOIN ingestion_jobs j ON j.generation_id = g.id
         WHERE j.document_id = $1
         ORDER BY j.created_at DESC,j.id DESC LIMIT 1`, [id],
      );
      if (!failed[0] || failed[0].latest_status !== 'failed') {
        throw conflict('INVALID_STATE', 'The latest ingestion job is not failed.');
      }

      const generationId = randomUUID();
      const jobId = randomUUID();
      await manager.query(
        `INSERT INTO index_generations (id,document_id,mode,status,embedding_model,embedding_digest,embedding_dimension,config)
         VALUES ($1,$2,$3,'queued',$4,$5,$6,$7::jsonb)`,
        [generationId, id, failed[0].mode, failed[0].embedding_model, failed[0].embedding_digest, failed[0].embedding_dimension, JSON.stringify(failed[0].config)],
      );
      await manager.query(
        `INSERT INTO ingestion_jobs (id,document_id,generation_id,status,stage)
         VALUES ($1,$2,$3,'queued','queued')`, [jobId, id, generationId],
      );
      if (document.status === 'failed') {
        await manager.query(`UPDATE documents SET status = 'queued',updated_at = now() WHERE id = $1`, [id]);
      }
    });
    return this.get(id);
  }

  async reindex(id: string): Promise<PublicDocument> {
    await this.dataSource.transaction(async (manager) => {
      const documents = await manager.query<{ id: string; status: string; deleted_at: Date | null }[]>(
        `SELECT id,status,deleted_at FROM documents WHERE id = $1 FOR UPDATE`, [id],
      );
      const document = documents[0];
      if (!document) throw notFound('Document not found.');
      if (document.deleted_at || document.status === 'deleted') throw conflict('INVALID_STATE', 'A deleted document cannot be reindexed.');
      if (document.status !== 'ready') throw conflict('DOCUMENT_NOT_READY', 'Only ready documents can be reindexed.');
      const active = await manager.query<{ id: string }[]>(
        `SELECT id FROM ingestion_jobs WHERE document_id = $1 AND status IN ('queued','processing') LIMIT 1`, [id],
      );
      if (active[0]) return;
      const artifact = this.indexingArtifact();
      await this.insertGenerationAndJob(manager, {
        documentId: id,
        generationId: randomUUID(),
        jobId: randomUUID(),
        artifact,
        mode: this.config.indexMode,
      });
    });
    return this.get(id);
  }

  async delete(id: string): Promise<void> {
    await this.dataSource.transaction(async (manager) => {
      const documents = await manager.query<{ id: string; status: string; active_generation_id: string | null; deleted_at: Date | null }[]>(
        `SELECT id,status,active_generation_id,deleted_at FROM documents WHERE id = $1 FOR UPDATE`, [id],
      );
      const document = documents[0];
      if (!document) throw notFound('Document not found.');
      if (document.deleted_at) return;

      const jobs = await manager.query<{ id: string; generation_id: string }[]>(
        `SELECT id,generation_id FROM ingestion_jobs
         WHERE document_id = $1 AND status IN ('queued','processing')
         ORDER BY id FOR UPDATE`, [id],
      );
      if (jobs.length) {
        await manager.query(
          `UPDATE ingestion_jobs SET status = 'cancelled',lease_token = NULL,lease_expires_at = NULL,updated_at = now()
           WHERE id = ANY($1::uuid[])`, [jobs.map((job) => job.id)],
        );
        await manager.query(
          `UPDATE index_generations SET status = 'cancelled',updated_at = now() WHERE id = ANY($1::uuid[])`,
          [jobs.map((job) => job.generation_id)],
        );
      }
      await manager.query(
        `UPDATE documents SET status = 'deleted',deleted_at = now(),active_generation_id = NULL,updated_at = now()
         WHERE id = $1`, [id],
      );
    });
  }

  async getSource(documentId: string, chunkId: string): Promise<Record<string, unknown>> {
    return this.dataSource.transaction(async (manager) => {
      const documents = await manager.query<{ id: string; name: string; deleted_at: Date | null }[]>(
        `SELECT id,name,deleted_at FROM documents WHERE id = $1 FOR SHARE`, [documentId],
      );
      const document = documents[0];
      if (!document) throw notFound('Source not found.');
      if (document.deleted_at) throw new ApiError('SOURCE_DELETED', 'This source has been deleted.', 410);

      const rows = await manager.query<{
        generation_id: string; page_start: number | null; page_end: number | null; heading: string | null; original_text: string;
      }[]>(
        `SELECT g.id AS generation_id,c.page_start,c.page_end,c.heading,c.original_text
         FROM chunks c JOIN index_generations g ON g.id = c.generation_id
         WHERE c.id = $1 AND g.document_id = $2 AND g.status = 'ready' LIMIT 1`, [chunkId, documentId],
      );
      if (!rows[0]) throw notFound('Source not found.');
      return {
        documentId: document.id,
        generationId: rows[0].generation_id,
        chunkId,
        documentName: document.name,
        pageStart: rows[0].page_start,
        pageEnd: rows[0].page_end,
        heading: rows[0].heading,
        text: rows[0].original_text,
      };
    });
  }

  private async insertGenerationAndJob(
    manager: { query: <T = unknown>(sql: string, parameters?: unknown[]) => Promise<T> },
    input: { documentId: string; generationId: string; jobId: string; artifact: IndexingArtifact; mode: string },
  ): Promise<void> {
    const { artifact } = input;
    const config = {
      chunkTargetTokens: artifact.chunkTargetTokens,
      chunkOverlapTokens: artifact.chunkOverlapTokens,
      chunkerVersion: artifact.chunkerVersion,
      tokenizerVersion: artifact.tokenizerVersion,
      contextPromptVersion: input.mode === 'contextual' ? artifact.contextPromptVersion : null,
      contextModel: input.mode === 'contextual' ? artifact.contextModel : null,
      contextModelDigest: input.mode === 'contextual' ? artifact.contextModelDigest : null,
      contextWindowTokens: artifact.contextWindowTokens,
      maxOutputTokens: artifact.maxOutputTokens,
    };
    await manager.query(
      `INSERT INTO index_generations (id,document_id,mode,status,embedding_model,embedding_digest,embedding_dimension,config)
       VALUES ($1,$2,$3,'queued',$4,$5,$6,$7::jsonb)`,
      [input.generationId, input.documentId, input.mode, artifact.embeddingModel, artifact.embeddingDigest, artifact.embeddingDimension, JSON.stringify(config)],
    );
    await manager.query(
      `INSERT INTO ingestion_jobs (id,document_id,generation_id,status,stage)
       VALUES ($1,$2,$3,'queued','queued')`, [input.jobId, input.documentId, input.generationId],
    );
  }

  private inspectUpload(file: Express.Multer.File): UploadMetadata {
    const normalizedName = (file.originalname ?? '').replace(/\\/g, '/').split('/').at(-1)?.trim() ?? '';
    if (!normalizedName || normalizedName.length > 255) throw badRequest('The file name must be between 1 and 255 characters.');
    const extension = extname(normalizedName).toLowerCase();
    const mime = (file.mimetype ?? '').toLowerCase();
    if (extension === '.pdf') {
      if (file.buffer.subarray(0, 5).toString('ascii') !== '%PDF-') {
        throw new ApiError('UNSUPPORTED_FILE_TYPE', 'The uploaded file is not a valid PDF.', 415);
      }
      return { displayName: normalizedName, extension: '.pdf', mimeType: 'application/pdf' };
    }
    if (extension !== '.txt' && extension !== '.md' && extension !== '.markdown') {
      throw new ApiError('UNSUPPORTED_FILE_TYPE', 'Only text PDFs, UTF-8 TXT, and Markdown files are supported.', 415);
    }
    if (!['text/plain', 'text/markdown', 'application/octet-stream', ''].includes(mime)) {
      throw new ApiError('UNSUPPORTED_FILE_TYPE', 'Only text PDFs, UTF-8 TXT, and Markdown files are supported.', 415);
    }
    try {
      const text = new TextDecoder('utf-8', { fatal: true }).decode(file.buffer);
      if (text.includes('\u0000')) throw new Error('Binary text content');
    } catch {
      throw new ApiError('UNSUPPORTED_FILE_TYPE', 'TXT and Markdown files must contain valid UTF-8 text.', 415);
    }
    const markdown = extension !== '.txt';
    return {
      displayName: normalizedName,
      extension: markdown ? '.md' : '.txt',
      mimeType: markdown ? 'text/markdown' : 'text/plain',
    };
  }

  private indexingArtifact(): IndexingArtifact {
    try {
      return readIndexingArtifact(this.config);
    } catch {
      throw unavailable('MODEL_UNAVAILABLE', 'The shared indexing configuration is missing or invalid.');
    }
  }
}

@Injectable()
export class DocumentCleanupService implements OnModuleInit, OnApplicationShutdown {
  private readonly logger = new Logger(DocumentCleanupService.name);
  private timer?: NodeJS.Timeout;
  private sweeping = false;

  constructor(
    @Inject(DATA_SOURCE) private readonly dataSource: DataSource,
    private readonly storage: LocalFileStorage,
  ) {}

  onModuleInit(): void {
    this.timer = setInterval(() => void this.sweep(), 30_000);
    this.timer.unref();
    void this.sweep();
  }

  onApplicationShutdown(): void {
    if (this.timer) clearInterval(this.timer);
  }

  async sweep(): Promise<void> {
    if (this.sweeping) return;
    this.sweeping = true;
    try {
      const rows = await this.dataSource.query<{ id: string; storage_key: string }[]>(
        `SELECT id,storage_key FROM documents
         WHERE deleted_at IS NOT NULL AND cleanup_completed_at IS NULL
         ORDER BY deleted_at,id LIMIT 20`,
      );
      for (const row of rows) {
        try {
          await this.storage.remove(row.storage_key);
          await this.cleanupIndex(row.id);
        } catch (error) {
          this.logger.warn(`Document cleanup will retry for ${row.id}: ${error instanceof Error ? error.message : 'storage/database error'}`);
        }
      }
      await this.cleanupOrphanedUploads();
    } catch (error) {
      this.logger.warn(`Document cleanup sweep failed: ${error instanceof Error ? error.message : 'database error'}`);
    } finally {
      this.sweeping = false;
    }
  }

  private async cleanupIndex(documentId: string): Promise<void> {
    await this.dataSource.transaction(async (manager) => {
      const documents = await manager.query<{ id: string; deleted_at: Date | null; cleanup_completed_at: Date | null }[]>(
        `SELECT id,deleted_at,cleanup_completed_at FROM documents WHERE id = $1 FOR UPDATE`, [documentId],
      );
      const document = documents[0];
      if (!document || !document.deleted_at || document.cleanup_completed_at) return;
      const generations = await manager.query<{ id: string }[]>(
        `SELECT id FROM index_generations WHERE document_id = $1 ORDER BY id`, [documentId],
      );
      const generationIds = generations.map((generation) => generation.id);
      if (generationIds.length) {
        await manager.query(`DELETE FROM chunks WHERE generation_id = ANY($1::uuid[])`, [generationIds]);
        await manager.query(`DELETE FROM ingestion_jobs WHERE generation_id = ANY($1::uuid[])`, [generationIds]);
        await manager.query(`DELETE FROM index_generations WHERE id = ANY($1::uuid[])`, [generationIds]);
      }
      await manager.query(`UPDATE documents SET cleanup_completed_at = now(),updated_at = now() WHERE id = $1`, [documentId]);
    });
  }

  private async cleanupOrphanedUploads(): Promise<void> {
    const candidates = await this.storage.staleStorageKeys(60 * 60 * 1000);
    if (!candidates.length) return;
    const referenced = await this.dataSource.query<{ storage_key: string }[]>(
      `SELECT storage_key FROM documents WHERE storage_key = ANY($1::text[])`, [candidates],
    );
    const keys = new Set(referenced.map((row) => row.storage_key));
    for (const candidate of candidates) {
      if (!keys.has(candidate)) await this.storage.remove(candidate).catch(() => undefined);
    }
  }
}

function documentSelect(): string {
  return `SELECT d.id,d.name,d.mime_type,d.size_bytes,d.status,d.active_generation_id,d.created_at,d.updated_at,
    row_to_json(j) AS latest_job
    FROM documents d
    LEFT JOIN LATERAL (
      SELECT id,status,stage,processed_chunks,total_chunks,error_code,error_message
      FROM ingestion_jobs WHERE document_id = d.id
      ORDER BY created_at DESC,id DESC LIMIT 1
    ) j ON true`;
}

function mapDocument(row: DocumentRow): PublicDocument {
  const job = row.latest_job;
  return {
    id: row.id,
    name: row.name,
    mimeType: row.mime_type,
    sizeBytes: Number(row.size_bytes),
    status: row.status,
    activeGenerationId: row.active_generation_id,
    latestJob: job ? {
      id: job.id,
      status: job.status,
      stage: job.stage,
      processedChunks: Number(job.processed_chunks),
      totalChunks: job.total_chunks === null ? null : Number(job.total_chunks),
      error: job.error_code ? mapIngestionFailure(job.error_code) : null,
    } : null,
    createdAt: new Date(row.created_at).toISOString(),
    updatedAt: new Date(row.updated_at).toISOString(),
  };
}

function mapIngestionFailure(code: string): { code: string; message: string } {
  const safeFailures: Record<string, { code: string; message: string }> = {
    EMPTY_DOCUMENT: { code: 'EMPTY_DOCUMENT', message: 'No readable text was found in this document.' },
    UNSUPPORTED_FILE_TYPE: { code: 'UNSUPPORTED_FILE_TYPE', message: 'This file format is not supported.' },
    CONTEXT_LIMIT_EXCEEDED: { code: 'CONTEXT_LIMIT_EXCEEDED', message: 'This document exceeds the configured model context limit.' },
    MODEL_UNAVAILABLE: { code: 'MODEL_UNAVAILABLE', message: 'The model service is unavailable.' },
    CORPUS_LIMIT_EXCEEDED: { code: 'CORPUS_LIMIT_EXCEEDED', message: 'The indexing capacity limit has been reached.' },
  };
  return safeFailures[code] ?? { code: 'INGESTION_FAILED', message: 'Document ingestion failed. You can retry.' };
}

function isUniqueViolation(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && (error as { code?: unknown }).code === '23505';
}
