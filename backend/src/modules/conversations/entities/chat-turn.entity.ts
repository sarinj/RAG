import { Column, CreateDateColumn, Entity, Index, PrimaryColumn, UpdateDateColumn } from 'typeorm';

@Entity({ name: 'chat_turns' })
@Index('chat_turns_request_uq', ['conversationId', 'requestId'], { unique: true })
@Index('chat_turns_pending_conversation_uq', ['conversationId'], { unique: true, where: "status = 'pending'" })
@Index('chat_turns_expiry_idx', ['status', 'expiresAt'])
export class ChatTurnEntity {
  @PrimaryColumn('uuid')
  id!: string;

  @Column('uuid', { name: 'conversation_id' })
  conversationId!: string;

  @Column('uuid', { name: 'request_id' })
  requestId!: string;

  @Column('text')
  question!: string;

  @Column('jsonb')
  sources!: { documentId: string; generationId: string }[];

  @Column('text')
  status!: string;

  @Column('timestamptz', { name: 'expires_at' })
  expiresAt!: Date;

  @CreateDateColumn({ name: 'created_at', type: 'timestamptz' })
  createdAt!: Date;

  @UpdateDateColumn({ name: 'updated_at', type: 'timestamptz' })
  updatedAt!: Date;
}
