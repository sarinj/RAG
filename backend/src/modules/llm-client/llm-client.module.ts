import { Module } from '@nestjs/common';
import { LlmClientService } from './llm-client.service';

@Module({ providers: [LlmClientService], exports: [LlmClientService] })
export class LlmClientModule {}
