import { Module } from '@nestjs/common';
import { LlmClientModule } from '../llm-client/llm-client.module';
import { ConversationsController } from './conversations.controller';
import { ConversationsService } from './conversations.service';

@Module({
  imports: [LlmClientModule],
  controllers: [ConversationsController],
  providers: [ConversationsService],
})
export class ConversationsModule {}
