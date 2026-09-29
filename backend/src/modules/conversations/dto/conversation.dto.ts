import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { ApiFailureDto } from '../../documents/dto/document.dto';

export class ScopeDto {
  @ApiProperty({ enum: ['all', 'selected'] }) mode!: string;
  @ApiPropertyOptional({ type: [String], format: 'uuid' }) documentIds?: string[];
}

export class ConversationDto {
  @ApiProperty({ format: 'uuid' }) id!: string;
  @ApiProperty() title!: string;
  @ApiProperty({ type: ScopeDto }) scope!: ScopeDto;
  @ApiProperty({ format: 'date-time' }) createdAt!: string;
  @ApiProperty({ format: 'date-time' }) updatedAt!: string;
}

export class CitationDto {
  @ApiProperty() label!: string;
  @ApiProperty({ format: 'uuid' }) documentId!: string;
  @ApiProperty({ format: 'uuid' }) generationId!: string;
  @ApiProperty({ format: 'uuid' }) chunkId!: string;
  @ApiProperty() documentName!: string;
  @ApiProperty({ nullable: true }) pageStart!: number | null;
  @ApiProperty({ nullable: true }) pageEnd!: number | null;
  @ApiProperty({ nullable: true }) heading!: string | null;
}

export class MessageDto {
  @ApiProperty({ format: 'uuid' }) id!: string;
  @ApiProperty({ format: 'uuid' }) conversationId!: string;
  @ApiProperty({ format: 'uuid' }) requestId!: string;
  @ApiProperty({ enum: ['user', 'assistant'] }) role!: string;
  @ApiProperty({ enum: ['pending', 'completed', 'failed'] }) status!: string;
  @ApiProperty() content!: string;
  @ApiProperty({ enum: ['answered', 'insufficient_evidence'], nullable: true }) answerStatus!: string | null;
  @ApiProperty({ type: [CitationDto] }) citations!: CitationDto[];
  @ApiProperty({ type: ApiFailureDto, nullable: true }) error!: ApiFailureDto | null;
  @ApiProperty({ format: 'date-time' }) createdAt!: string;
  @ApiProperty({ format: 'date-time' }) updatedAt!: string;
}

export class TurnDto {
  @ApiProperty({ format: 'uuid' }) requestId!: string;
  @ApiProperty({ type: MessageDto }) userMessage!: MessageDto;
  @ApiProperty({ type: MessageDto }) assistantMessage!: MessageDto;
}

export class ConversationPageDto {
  @ApiProperty({ type: [ConversationDto] }) items!: ConversationDto[];
  @ApiProperty({ nullable: true }) nextCursor!: string | null;
}

export class MessagePageDto {
  @ApiProperty({ type: [MessageDto] }) items!: MessageDto[];
  @ApiProperty({ nullable: true }) nextCursor!: string | null;
}
