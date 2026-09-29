import { Inject, Injectable } from '@nestjs/common';
import { DataSource } from 'typeorm';
import { CHUNK_CAPACITY_LIMIT, ServerConfig } from '../../config/configuration';
import { readIndexingArtifact } from '../../config/index-config';
import { DATA_SOURCE, SERVER_CONFIG } from '../../config/tokens';
import { LocalFileStorage } from '../storage/storage.service';
import { LlmClientService } from '../llm-client/llm-client.service';

const REQUIRED_COLUMNS = [
  ['documents', 'id'], ['documents', 'name'], ['documents', 'mime_type'], ['documents', 'size_bytes'],
  ['documents', 'content_hash'], ['documents', 'storage_key'], ['documents', 'status'],
  ['documents', 'active_generation_id'], ['documents', 'created_at'], ['documents', 'updated_at'],
  ['documents', 'deleted_at'], ['documents', 'cleanup_completed_at'],
  ['index_generations', 'id'], ['index_generations', 'document_id'], ['index_generations', 'mode'],
  ['index_generations', 'status'], ['index_generations', 'embedding_model'], ['index_generations', 'embedding_digest'],
  ['index_generations', 'embedding_dimension'], ['index_generations', 'config'],
  ['index_generations', 'created_at'], ['index_generations', 'updated_at'],
  ['ingestion_jobs', 'id'], ['ingestion_jobs', 'document_id'], ['ingestion_jobs', 'generation_id'],
  ['ingestion_jobs', 'status'], ['ingestion_jobs', 'stage'], ['ingestion_jobs', 'processed_chunks'],
  ['ingestion_jobs', 'total_chunks'], ['ingestion_jobs', 'attempts'], ['ingestion_jobs', 'lease_token'],
  ['ingestion_jobs', 'lease_expires_at'], ['ingestion_jobs', 'heartbeat_at'], ['ingestion_jobs', 'error_code'],
  ['ingestion_jobs', 'error_message'], ['ingestion_jobs', 'created_at'], ['ingestion_jobs', 'updated_at'],
  ['chunks', 'id'], ['chunks', 'generation_id'], ['chunks', 'ordinal'], ['chunks', 'original_text'],
  ['chunks', 'context_text'], ['chunks', 'searchable_text'], ['chunks', 'page_start'], ['chunks', 'page_end'],
  ['chunks', 'heading'], ['chunks', 'source_start'], ['chunks', 'source_end'], ['chunks', 'token_count'],
  ['chunks', 'embedding'], ['chunks', 'created_at'],
  ['conversations', 'id'], ['conversations', 'title'], ['conversations', 'scope_mode'],
  ['conversations', 'created_at'], ['conversations', 'updated_at'],
  ['conversation_documents', 'conversation_id'], ['conversation_documents', 'document_id'],
  ['chat_turns', 'id'], ['chat_turns', 'conversation_id'], ['chat_turns', 'request_id'],
  ['chat_turns', 'question'], ['chat_turns', 'sources'], ['chat_turns', 'status'], ['chat_turns', 'expires_at'],
  ['chat_turns', 'created_at'], ['chat_turns', 'updated_at'],
  ['messages', 'id'], ['messages', 'turn_id'], ['messages', 'conversation_id'], ['messages', 'role'],
  ['messages', 'status'], ['messages', 'content'], ['messages', 'answer_status'], ['messages', 'error_code'],
  ['messages', 'error_message'], ['messages', 'created_at'], ['messages', 'updated_at'],
  ['message_citations', 'message_id'], ['message_citations', 'label'], ['message_citations', 'document_id'],
  ['message_citations', 'generation_id'], ['message_citations', 'chunk_id'], ['message_citations', 'document_name'],
  ['message_citations', 'page_start'], ['message_citations', 'page_end'], ['message_citations', 'heading'],
] as const;

const REQUIRED_INDEXES = [
  'documents_active_content_hash_uq', 'ingestion_jobs_status_lease_idx', 'ingestion_jobs_document_active_uq',
  'chunks_generation_ordinal_uq', 'chunks_generation_idx', 'conversation_documents_document_idx',
  'chat_turns_request_uq', 'chat_turns_pending_conversation_uq', 'chat_turns_expiry_idx',
  'messages_turn_role_uq', 'messages_conversation_created_idx', 'message_citations_document_idx',
] as const;

@Injectable()
export class HealthService {
  constructor(
    @Inject(DATA_SOURCE) private readonly dataSource: DataSource,
    @Inject(SERVER_CONFIG) private readonly config: ServerConfig,
    private readonly storage: LocalFileStorage,
    private readonly llm: LlmClientService,
  ) {}

  async isReady(): Promise<boolean> {
    try {
      const artifact = readIndexingArtifact(this.config);
      if (!this.dataSource.isInitialized) return false;
      const [schemaRows, migrationRows, columnRows, indexRows, vectorRows, capacityRows, triggerRows] = await Promise.all([
        this.dataSource.query<Record<string, string | null>[]>(
          `SELECT to_regclass('public.documents')::text AS documents_exists,
                  to_regclass('public.chunks')::text AS chunks_exists,
                  to_regclass('public.index_generations')::text AS generations_exists,
                  to_regclass('public.ingestion_jobs')::text AS jobs_exists,
                  to_regclass('public.conversations')::text AS conversations_exists,
                  to_regclass('public.conversation_documents')::text AS conversation_documents_exists,
                  to_regclass('public.chat_turns')::text AS turns_exists,
                  to_regclass('public.messages')::text AS messages_exists,
                  to_regclass('public.message_citations')::text AS citations_exists,
                  to_regclass('public.typeorm_migrations')::text AS migrations_exists`,
        ),
        this.dataSource.query<{ migration_applied: boolean }[]>(
          `SELECT EXISTS (
             SELECT 1 FROM public.typeorm_migrations
             WHERE name = 'InitialSchema1710000000000'
           ) AS migration_applied`,
        ),
        this.dataSource.query<{ missing_columns: number }[]>(
          `SELECT count(*)::integer AS missing_columns
           FROM unnest($1::text[], $2::text[]) AS required(table_name, column_name)
           LEFT JOIN pg_class c ON c.relname = required.table_name
             AND c.relnamespace = 'public'::regnamespace
           LEFT JOIN pg_attribute a ON a.attrelid = c.oid AND a.attname = required.column_name
             AND a.attnum > 0 AND NOT a.attisdropped
           WHERE a.attnum IS NULL`,
          [REQUIRED_COLUMNS.map(([table]) => table), REQUIRED_COLUMNS.map(([, column]) => column)],
        ),
        this.dataSource.query<{ missing_indexes: number }[]>(
          `SELECT count(*)::integer AS missing_indexes
           FROM unnest($1::text[]) AS required(index_name)
           LEFT JOIN pg_class c ON c.relname = required.index_name
             AND c.relnamespace = 'public'::regnamespace AND c.relkind = 'i'
           LEFT JOIN pg_index i ON i.indexrelid = c.oid AND i.indisvalid AND i.indisready
           WHERE i.indexrelid IS NULL`,
          [REQUIRED_INDEXES],
        ),
        this.dataSource.query<{ embedding_type: string | null }[]>(
          `SELECT format_type(a.atttypid,a.atttypmod) AS embedding_type
           FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid
           WHERE c.relname = 'chunks' AND a.attname = 'embedding' AND a.attnum > 0 AND NOT a.attisdropped`,
        ),
        this.dataSource.query<{ max_chunks: number; reserve_function_exists: boolean }[]>(
          `SELECT rag_chunk_capacity_limit() AS max_chunks,
                  to_regprocedure('public.rag_reserve_chunk_capacity(integer)') IS NOT NULL AS reserve_function_exists`,
        ),
        this.dataSource.query<{ before_insert_enabled: boolean; after_insert_enabled: boolean }[]>(
          `SELECT
             EXISTS (
               SELECT 1 FROM pg_trigger t
               JOIN pg_class c ON c.oid = t.tgrelid
               JOIN pg_namespace n ON n.oid = c.relnamespace
               JOIN pg_proc p ON p.oid = t.tgfoid
               WHERE n.nspname = 'public' AND c.relname = 'chunks'
                 AND t.tgname = 'chunks_capacity_lock_before_insert'
                 AND p.proname = 'rag_lock_chunk_capacity_before_insert'
                 AND t.tgenabled IN ('O','A') AND NOT t.tgisinternal
             ) AS before_insert_enabled,
             EXISTS (
               SELECT 1 FROM pg_trigger t
               JOIN pg_class c ON c.oid = t.tgrelid
               JOIN pg_namespace n ON n.oid = c.relnamespace
               JOIN pg_proc p ON p.oid = t.tgfoid
               WHERE n.nspname = 'public' AND c.relname = 'chunks'
                 AND t.tgname = 'chunks_capacity_check_after_insert'
                 AND p.proname = 'rag_check_chunk_capacity_after_insert'
                 AND t.tgenabled IN ('O','A') AND NOT t.tgisinternal
             ) AS after_insert_enabled`,
        ),
      ]);
      if (!schemaRows[0] || Object.values(schemaRows[0]).some((value) => !value)) return false;
      if (!migrationRows[0]?.migration_applied) return false;
      if (Number(columnRows[0]?.missing_columns) !== 0) return false;
      if (Number(indexRows[0]?.missing_indexes) !== 0) return false;
      if (vectorRows[0]?.embedding_type !== `vector(${artifact.embeddingDimension})`) return false;
      if (Number(capacityRows[0]?.max_chunks) !== CHUNK_CAPACITY_LIMIT || !capacityRows[0]?.reserve_function_exists) return false;
      if (!triggerRows[0]?.before_insert_enabled || !triggerRows[0]?.after_insert_enabled) return false;
      const [storageReady, llmReady] = await Promise.all([this.storage.isWritable(), this.llm.ready()]);
      return storageReady && llmReady;
    } catch {
      return false;
    }
  }
}
