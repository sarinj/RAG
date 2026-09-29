import { Column, CreateDateColumn, Entity, Index, PrimaryColumn, UpdateDateColumn } from 'typeorm';

@Entity({ name: 'messages' })
@Index('messages_turn_role_uq', ['turnId', 'role'], { unique: true })
@Index('messages_conversation_created_idx', ['conversationId', 'createdAt', 'id'])
export class MessageEntity {
  @PrimaryColumn('uuid')
  id!: string;

  @Column('uuid', { name: 'turn_id' })
  turnId!: string;

  @Column('uuid', { name: 'conversation_id' })
  conversationId!: string;

  @Column('text')
  role!: string;

  @Column('text')
  status!: string;

  @Column('text')
  content!: string;

  @Column('text', { name: 'answer_status', nullable: true })
  answerStatus!: string | null;

  @Column('text', { name: 'error_code', nullable: true })
  errorCode!: string | null;

  @Column('text', { name: 'error_message', nullable: true })
  errorMessage!: string | null;

  @CreateDateColumn({ name: 'created_at', type: 'timestamptz' })
  createdAt!: Date;

  @UpdateDateColumn({ name: 'updated_at', type: 'timestamptz' })
  updatedAt!: Date;
}
