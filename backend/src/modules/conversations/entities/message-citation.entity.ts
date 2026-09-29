import { Column, Entity, Index, PrimaryColumn } from 'typeorm';

@Entity({ name: 'message_citations' })
@Index('message_citations_document_idx', ['documentId'])
export class MessageCitationEntity {
  @PrimaryColumn('uuid', { name: 'message_id' })
  messageId!: string;

  @PrimaryColumn('text')
  label!: string;

  @Column('uuid', { name: 'document_id' })
  documentId!: string;

  @Column('uuid', { name: 'generation_id' })
  generationId!: string;

  @Column('uuid', { name: 'chunk_id' })
  chunkId!: string;

  @Column('text', { name: 'document_name' })
  documentName!: string;

  @Column('integer', { name: 'page_start', nullable: true })
  pageStart!: number | null;

  @Column('integer', { name: 'page_end', nullable: true })
  pageEnd!: number | null;

  @Column('text', { nullable: true })
  heading!: string | null;
}
