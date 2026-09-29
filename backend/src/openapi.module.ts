import { Module } from '@nestjs/common';
import { ConversationsController } from './modules/conversations/conversations.controller';
import { ConversationsService } from './modules/conversations/conversations.service';
import { DocumentsController } from './modules/documents/documents.controller';
import { DocumentsService, DocumentCleanupService } from './modules/documents/documents.service';
import { HealthController } from './modules/health/health.controller';
import { HealthService } from './modules/health/health.service';

@Module({
  controllers: [DocumentsController, ConversationsController, HealthController],
  providers: [
    { provide: DocumentsService, useValue: {} },
    { provide: DocumentCleanupService, useValue: {} },
    { provide: ConversationsService, useValue: {} },
    { provide: HealthService, useValue: {} },
  ],
})
export class OpenApiModule {}
