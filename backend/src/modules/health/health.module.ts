import { Module } from '@nestjs/common';
import { LlmClientModule } from '../llm-client/llm-client.module';
import { StorageModule } from '../storage/storage.module';
import { HealthController } from './health.controller';
import { HealthService } from './health.service';

@Module({
  imports: [LlmClientModule, StorageModule],
  controllers: [HealthController],
  providers: [HealthService],
})
export class HealthModule {}
