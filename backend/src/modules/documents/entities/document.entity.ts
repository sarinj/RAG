import { Column, CreateDateColumn, Entity, Index, PrimaryColumn, UpdateDateColumn } from 'typeorm';

@Entity({ name: 'documents' })
@Index('documents_active_content_hash_uq', ['contentHash'], { unique: true, where: 'deleted_at IS NULL' })
export class DocumentEntity {
  @PrimaryColumn('uuid')
  id!: string;

  @Column('text')
  name!: string;

  @Column('text', { name: 'mime_type' })
  mimeType!: string;

  @Column('bigint', { name: 'size_bytes' })
  sizeBytes!: string;

  @Column('text', { name: 'content_hash' })
  contentHash!: string;

  @Column('text', { name: 'storage_key' })
  storageKey!: string;

  @Column('text')
  status!: string;

  @Column('uuid', { name: 'active_generation_id', nullable: true })
  activeGenerationId!: string | null;

  @CreateDateColumn({ name: 'created_at', type: 'timestamptz' })
  createdAt!: Date;

  @UpdateDateColumn({ name: 'updated_at', type: 'timestamptz' })
  updatedAt!: Date;

  @Column('timestamptz', { name: 'deleted_at', nullable: true })
  deletedAt!: Date | null;

  @Column('timestamptz', { name: 'cleanup_completed_at', nullable: true })
  cleanupCompletedAt!: Date | null;
}
