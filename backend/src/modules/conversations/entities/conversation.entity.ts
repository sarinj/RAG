import { Column, CreateDateColumn, Entity, PrimaryColumn, UpdateDateColumn } from 'typeorm';

@Entity({ name: 'conversations' })
export class ConversationEntity {
  @PrimaryColumn('uuid')
  id!: string;

  @Column('text')
  title!: string;

  @Column('text', { name: 'scope_mode' })
  scopeMode!: string;

  @CreateDateColumn({ name: 'created_at', type: 'timestamptz' })
  createdAt!: Date;

  @UpdateDateColumn({ name: 'updated_at', type: 'timestamptz' })
  updatedAt!: Date;
}
