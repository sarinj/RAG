import 'dotenv/config';
import { DataSource, DataSourceOptions } from 'typeorm';
import { parseServerConfig } from '../config/configuration';
import { ChunkEntity } from '../modules/ingestion/entities/chunk.entity';
import { IndexGenerationEntity } from '../modules/ingestion/entities/index-generation.entity';
import { IngestionJobEntity } from '../modules/ingestion/entities/ingestion-job.entity';
import { ConversationEntity } from '../modules/conversations/entities/conversation.entity';
import { ConversationDocumentEntity } from '../modules/conversations/entities/conversation-document.entity';
import { ChatTurnEntity } from '../modules/conversations/entities/chat-turn.entity';
import { MessageEntity } from '../modules/conversations/entities/message.entity';
import { MessageCitationEntity } from '../modules/conversations/entities/message-citation.entity';
import { DocumentEntity } from '../modules/documents/entities/document.entity';
import { InitialSchema1710000000000 } from './migrations/1710000000000-InitialSchema';
import { SmallTalkAnswerStatus1710000001000 } from './migrations/1710000001000-SmallTalkAnswerStatus';
import { DirectAnswerStatus1710000002000 } from './migrations/1710000002000-DirectAnswerStatus';

export function createDataSourceOptions(): DataSourceOptions {
  const config = parseServerConfig();
  return {
    type: 'postgres',
    url: config.databaseUrl,
    entities: [
      DocumentEntity,
      IndexGenerationEntity,
      IngestionJobEntity,
      ChunkEntity,
      ConversationEntity,
      ConversationDocumentEntity,
      ChatTurnEntity,
      MessageEntity,
      MessageCitationEntity,
    ],
    migrations: [
      InitialSchema1710000000000,
      SmallTalkAnswerStatus1710000001000,
      DirectAnswerStatus1710000002000,
    ],
    migrationsTableName: 'typeorm_migrations',
    migrationsRun: false,
    synchronize: false,
    logging: false,
    extra: { application_name: 'learning-rag-backend' },
  };
}

export const AppDataSource = new DataSource(createDataSourceOptions());
