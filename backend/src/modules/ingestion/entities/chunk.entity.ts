import { Column, CreateDateColumn, Entity, Index, PrimaryColumn } from 'typeorm';

@Entity({ name: 'chunks' })
@Index('chunks_generation_ordinal_uq', ['generationId', 'ordinal'], { unique: true })
@Index('chunks_generation_idx', ['generationId'])
export class ChunkEntity {
  @PrimaryColumn('uuid')
  id!: string;

  @Column('uuid', { name: 'generation_id' })
  generationId!: string;

  @Column('integer')
  ordinal!: number;

  @Column('text', { name: 'original_text' })
  originalText!: string;

  @Column('text', { name: 'context_text', nullable: true })
  contextText!: string | null;

  @Column('text', { name: 'searchable_text' })
  searchableText!: string;

  @Column('integer', { name: 'page_start', nullable: true })
  pageStart!: number | null;

  @Column('integer', { name: 'page_end', nullable: true })
  pageEnd!: number | null;

  @Column('text', { nullable: true })
  heading!: string | null;

  @Column('integer', { name: 'source_start' })
  sourceStart!: number;

  @Column('integer', { name: 'source_end' })
  sourceEnd!: number;

  @Column('integer', { name: 'token_count' })
  tokenCount!: number;

  @Column('vector', { select: false })
  embedding!: number[];

  @CreateDateColumn({ name: 'created_at', type: 'timestamptz' })
  createdAt!: Date;
}
