import { Column, CreateDateColumn, Entity, Index, PrimaryColumn, UpdateDateColumn } from 'typeorm';

@Entity({ name: 'index_generations' })
@Index('index_generations_document_id_id_uq', ['documentId', 'id'], { unique: true })
export class IndexGenerationEntity {
  @PrimaryColumn('uuid')
  id!: string;

  @Column('uuid', { name: 'document_id' })
  documentId!: string;

  @Column('text')
  mode!: string;

  @Column('text')
  status!: string;

  @Column('text', { name: 'embedding_model' })
  embeddingModel!: string;

  @Column('text', { name: 'embedding_digest' })
  embeddingDigest!: string;

  @Column('integer', { name: 'embedding_dimension' })
  embeddingDimension!: number;

  @Column('jsonb')
  config!: Record<string, unknown>;

  @CreateDateColumn({ name: 'created_at', type: 'timestamptz' })
  createdAt!: Date;

  @UpdateDateColumn({ name: 'updated_at', type: 'timestamptz' })
  updatedAt!: Date;
}
