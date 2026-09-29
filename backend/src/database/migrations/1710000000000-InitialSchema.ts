import { MigrationInterface, QueryRunner } from 'typeorm';
import { CHUNK_CAPACITY_LIMIT, parseServerConfig } from '../../config/configuration';
import { readIndexingArtifact } from '../../config/index-config';

export class InitialSchema1710000000000 implements MigrationInterface {
  name = 'InitialSchema1710000000000';

  async up(queryRunner: QueryRunner): Promise<void> {
    const config = parseServerConfig();
    const { embeddingDimension } = readIndexingArtifact(config);
    const vectorDimension = String(embeddingDimension);

    await queryRunner.query('CREATE EXTENSION IF NOT EXISTS vector');
    await queryRunner.query(`
      CREATE TABLE documents (
        id uuid PRIMARY KEY,
        name text NOT NULL,
        mime_type text NOT NULL,
        size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
        content_hash text NOT NULL,
        storage_key text NOT NULL,
        status text NOT NULL CHECK (status IN ('queued','processing','ready','failed','deleted')),
        active_generation_id uuid NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        deleted_at timestamptz NULL,
        cleanup_completed_at timestamptz NULL,
        CONSTRAINT documents_deleted_state_ck CHECK ((status = 'deleted') = (deleted_at IS NOT NULL))
      )
    `);
    await queryRunner.query(`
      CREATE UNIQUE INDEX documents_active_content_hash_uq
      ON documents (content_hash) WHERE deleted_at IS NULL
    `);
    await queryRunner.query(`
      CREATE TABLE index_generations (
        id uuid PRIMARY KEY,
        document_id uuid NOT NULL REFERENCES documents(id) ON DELETE RESTRICT,
        mode text NOT NULL CHECK (mode IN ('baseline','contextual')),
        status text NOT NULL CHECK (status IN ('queued','processing','ready','failed','cancelled')),
        embedding_model text NOT NULL,
        embedding_digest text NOT NULL,
        embedding_dimension integer NOT NULL CHECK (embedding_dimension = ${vectorDimension}),
        config jsonb NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT index_generations_document_id_id_uq UNIQUE (document_id,id)
      )
    `);
    await queryRunner.query(`
      ALTER TABLE documents
      ADD CONSTRAINT documents_active_generation_owner_fk
      FOREIGN KEY (id,active_generation_id)
      REFERENCES index_generations(document_id,id) ON DELETE RESTRICT
    `);
    await queryRunner.query(`
      CREATE TABLE ingestion_jobs (
        id uuid PRIMARY KEY,
        document_id uuid NOT NULL REFERENCES documents(id) ON DELETE RESTRICT,
        generation_id uuid NOT NULL UNIQUE,
        status text NOT NULL CHECK (status IN ('queued','processing','succeeded','failed','cancelled')),
        stage text NOT NULL CHECK (stage IN ('queued','parsing','chunking','contextualizing','embedding','publishing','done')),
        processed_chunks integer NOT NULL DEFAULT 0 CHECK (processed_chunks >= 0),
        total_chunks integer NULL CHECK (total_chunks IS NULL OR total_chunks >= 0),
        attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
        lease_token uuid NULL,
        lease_expires_at timestamptz NULL,
        heartbeat_at timestamptz NULL,
        error_code text NULL,
        error_message text NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT ingestion_jobs_generation_owner_fk
          FOREIGN KEY (document_id,generation_id)
          REFERENCES index_generations(document_id,id) ON DELETE RESTRICT
      )
    `);
    await queryRunner.query(`
      CREATE INDEX ingestion_jobs_status_lease_idx ON ingestion_jobs(status,lease_expires_at)
    `);
    await queryRunner.query(`
      CREATE UNIQUE INDEX ingestion_jobs_document_active_uq
      ON ingestion_jobs(document_id) WHERE status IN ('queued','processing')
    `);
    await queryRunner.query(`
      CREATE TABLE chunks (
        id uuid PRIMARY KEY,
        generation_id uuid NOT NULL REFERENCES index_generations(id) ON DELETE RESTRICT,
        ordinal integer NOT NULL CHECK (ordinal >= 0),
        original_text text NOT NULL,
        context_text text NULL,
        searchable_text text NOT NULL,
        page_start integer NULL,
        page_end integer NULL,
        heading text NULL,
        source_start integer NOT NULL CHECK (source_start >= 0),
        source_end integer NOT NULL CHECK (source_end >= source_start),
        token_count integer NOT NULL CHECK (token_count >= 0),
        embedding vector(${vectorDimension}) NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chunks_page_range_ck CHECK (
          (page_start IS NULL OR page_start >= 1) AND
          (page_end IS NULL OR page_end >= 1) AND
          (page_start IS NULL OR page_end IS NULL OR page_start <= page_end)
        ),
        CONSTRAINT chunks_generation_ordinal_uq UNIQUE (generation_id,ordinal)
      )
    `);
    await queryRunner.query('CREATE INDEX chunks_generation_idx ON chunks(generation_id)');
    await queryRunner.query(`
      CREATE FUNCTION rag_chunk_capacity_limit() RETURNS integer
      LANGUAGE sql IMMUTABLE PARALLEL SAFE
      AS $$ SELECT ${CHUNK_CAPACITY_LIMIT}::integer $$
    `);
    await queryRunner.query(`
      CREATE FUNCTION rag_reserve_chunk_capacity(batch_size integer) RETURNS void
      LANGUAGE plpgsql
      AS $$
      DECLARE stored_chunks bigint;
      BEGIN
        IF batch_size IS NULL OR batch_size < 0 THEN
          RAISE EXCEPTION USING ERRCODE = 'P0001', MESSAGE = 'CORPUS_LIMIT_EXCEEDED';
        END IF;
        PERFORM pg_advisory_xact_lock(71001);
        SELECT count(*) INTO stored_chunks FROM chunks;
        IF stored_chunks + batch_size > rag_chunk_capacity_limit() THEN
          RAISE EXCEPTION USING ERRCODE = 'P0001', MESSAGE = 'CORPUS_LIMIT_EXCEEDED';
        END IF;
      END;
      $$
    `);
    await queryRunner.query(`
      CREATE FUNCTION rag_lock_chunk_capacity_before_insert() RETURNS trigger
      LANGUAGE plpgsql
      AS $$
      BEGIN
        PERFORM rag_reserve_chunk_capacity(0);
        RETURN NULL;
      END;
      $$
    `);
    await queryRunner.query(`
      CREATE FUNCTION rag_check_chunk_capacity_after_insert() RETURNS trigger
      LANGUAGE plpgsql
      AS $$
      BEGIN
        PERFORM rag_reserve_chunk_capacity(0);
        RETURN NULL;
      END;
      $$
    `);
    await queryRunner.query(`
      CREATE TRIGGER chunks_capacity_lock_before_insert
      BEFORE INSERT ON chunks FOR EACH STATEMENT
      EXECUTE FUNCTION rag_lock_chunk_capacity_before_insert()
    `);
    await queryRunner.query(`
      CREATE TRIGGER chunks_capacity_check_after_insert
      AFTER INSERT ON chunks FOR EACH STATEMENT
      EXECUTE FUNCTION rag_check_chunk_capacity_after_insert()
    `);
    await queryRunner.query(`
      CREATE TABLE conversations (
        id uuid PRIMARY KEY,
        title text NOT NULL CHECK (char_length(title) BETWEEN 1 AND 120),
        scope_mode text NOT NULL CHECK (scope_mode IN ('all','selected')),
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now()
      )
    `);
    await queryRunner.query(`
      CREATE TABLE conversation_documents (
        conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE RESTRICT,
        document_id uuid NOT NULL REFERENCES documents(id) ON DELETE RESTRICT,
        PRIMARY KEY (conversation_id,document_id)
      )
    `);
    await queryRunner.query('CREATE INDEX conversation_documents_document_idx ON conversation_documents(document_id)');
    await queryRunner.query(`
      CREATE TABLE chat_turns (
        id uuid PRIMARY KEY,
        conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE RESTRICT,
        request_id uuid NOT NULL,
        question text NOT NULL,
        sources jsonb NOT NULL CHECK (jsonb_typeof(sources) = 'array'),
        status text NOT NULL CHECK (status IN ('pending','completed','failed')),
        expires_at timestamptz NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chat_turns_request_uq UNIQUE (conversation_id,request_id)
      )
    `);
    await queryRunner.query(`
      CREATE UNIQUE INDEX chat_turns_pending_conversation_uq
      ON chat_turns(conversation_id) WHERE status = 'pending'
    `);
    await queryRunner.query('CREATE INDEX chat_turns_expiry_idx ON chat_turns(status,expires_at)');
    await queryRunner.query(`
      CREATE TABLE messages (
        id uuid PRIMARY KEY,
        turn_id uuid NOT NULL REFERENCES chat_turns(id) ON DELETE RESTRICT,
        conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE RESTRICT,
        role text NOT NULL CHECK (role IN ('user','assistant')),
        status text NOT NULL CHECK (status IN ('pending','completed','failed')),
        content text NOT NULL,
        answer_status text NULL CHECK (answer_status IS NULL OR answer_status IN ('answered','insufficient_evidence')),
        error_code text NULL,
        error_message text NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT messages_turn_role_uq UNIQUE (turn_id,role)
      )
    `);
    await queryRunner.query('CREATE INDEX messages_conversation_created_idx ON messages(conversation_id,created_at,id)');
    await queryRunner.query(`
      CREATE TABLE message_citations (
        message_id uuid NOT NULL REFERENCES messages(id) ON DELETE RESTRICT,
        label text NOT NULL,
        document_id uuid NOT NULL REFERENCES documents(id) ON DELETE RESTRICT,
        generation_id uuid NOT NULL,
        chunk_id uuid NOT NULL,
        document_name text NOT NULL,
        page_start integer NULL,
        page_end integer NULL,
        heading text NULL,
        CONSTRAINT message_citations_page_range_ck CHECK (
          (page_start IS NULL OR page_start >= 1) AND
          (page_end IS NULL OR page_end >= 1) AND
          (page_start IS NULL OR page_end IS NULL OR page_start <= page_end)
        ),
        PRIMARY KEY (message_id,label)
      )
    `);
    await queryRunner.query('CREATE INDEX message_citations_document_idx ON message_citations(document_id)');
  }

  async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query('DROP TABLE message_citations');
    await queryRunner.query('DROP TABLE messages');
    await queryRunner.query('DROP TABLE chat_turns');
    await queryRunner.query('DROP TABLE conversation_documents');
    await queryRunner.query('DROP TABLE conversations');
    await queryRunner.query('DROP TABLE chunks');
    await queryRunner.query('DROP FUNCTION rag_check_chunk_capacity_after_insert()');
    await queryRunner.query('DROP FUNCTION rag_lock_chunk_capacity_before_insert()');
    await queryRunner.query('DROP FUNCTION rag_reserve_chunk_capacity(integer)');
    await queryRunner.query('DROP FUNCTION rag_chunk_capacity_limit()');
    await queryRunner.query('DROP TABLE ingestion_jobs');
    await queryRunner.query('ALTER TABLE documents DROP CONSTRAINT documents_active_generation_owner_fk');
    await queryRunner.query('DROP TABLE index_generations');
    await queryRunner.query('DROP TABLE documents');
    await queryRunner.query('DROP EXTENSION vector');
  }
}
