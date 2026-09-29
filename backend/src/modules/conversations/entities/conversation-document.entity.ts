import { Entity, Index, PrimaryColumn } from 'typeorm';

@Entity({ name: 'conversation_documents' })
@Index('conversation_documents_document_idx', ['documentId'])
export class ConversationDocumentEntity {
  @PrimaryColumn('uuid', { name: 'conversation_id' })
  conversationId!: string;

  @PrimaryColumn('uuid', { name: 'document_id' })
  documentId!: string;
}
