import { Module } from '@nestjs/common';
import { StorageModule } from '../storage/storage.module';
import { DocumentsController } from './documents.controller';
import { DocumentCleanupService, DocumentsService } from './documents.service';

@Module({
  imports: [StorageModule],
  controllers: [DocumentsController],
  providers: [DocumentsService, DocumentCleanupService],
  exports: [DocumentsService],
})
export class DocumentsModule {}
