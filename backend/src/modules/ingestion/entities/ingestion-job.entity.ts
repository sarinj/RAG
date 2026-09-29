import { Column, CreateDateColumn, Entity, Index, PrimaryColumn, UpdateDateColumn } from 'typeorm';

@Entity({ name: 'ingestion_jobs' })
@Index('ingestion_jobs_status_lease_idx', ['status', 'leaseExpiresAt'])
@Index('ingestion_jobs_document_active_uq', ['documentId'], {
  unique: true,
  where: "status IN ('queued','processing')",
})
export class IngestionJobEntity {
  @PrimaryColumn('uuid')
  id!: string;

  @Column('uuid', { name: 'document_id' })
  documentId!: string;

  @Column('uuid', { name: 'generation_id', unique: true })
  generationId!: string;

  @Column('text')
  status!: string;

  @Column('text')
  stage!: string;

  @Column('integer', { name: 'processed_chunks', default: 0 })
  processedChunks!: number;

  @Column('integer', { name: 'total_chunks', nullable: true })
  totalChunks!: number | null;

  @Column('integer', { default: 0 })
  attempts!: number;

  @Column('uuid', { name: 'lease_token', nullable: true })
  leaseToken!: string | null;

  @Column('timestamptz', { name: 'lease_expires_at', nullable: true })
  leaseExpiresAt!: Date | null;

  @Column('timestamptz', { name: 'heartbeat_at', nullable: true })
  heartbeatAt!: Date | null;

  @Column('text', { name: 'error_code', nullable: true })
  errorCode!: string | null;

  @Column('text', { name: 'error_message', nullable: true })
  errorMessage!: string | null;

  @CreateDateColumn({ name: 'created_at', type: 'timestamptz' })
  createdAt!: Date;

  @UpdateDateColumn({ name: 'updated_at', type: 'timestamptz' })
  updatedAt!: Date;
}
