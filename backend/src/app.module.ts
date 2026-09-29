import { Module } from '@nestjs/common';
import { DatabaseModule } from './database/database.module';
import { DocumentsModule } from './modules/documents/documents.module';
import { ConversationsModule } from './modules/conversations/conversations.module';
import { HealthModule } from './modules/health/health.module';

@Module({ imports: [DatabaseModule, DocumentsModule, ConversationsModule, HealthModule] })
export class AppModule {}
