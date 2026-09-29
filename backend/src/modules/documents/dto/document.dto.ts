import { ApiProperty } from '@nestjs/swagger';

export class ApiFailureDto {
  @ApiProperty() code!: string;
  @ApiProperty() message!: string;
}

export class JobDto {
  @ApiProperty({ format: 'uuid' }) id!: string;
  @ApiProperty({ enum: ['queued', 'processing', 'succeeded', 'failed', 'cancelled'] }) status!: string;
  @ApiProperty({ enum: ['queued', 'parsing', 'chunking', 'contextualizing', 'embedding', 'publishing', 'done'] }) stage!: string;
  @ApiProperty() processedChunks!: number;
  @ApiProperty({ nullable: true }) totalChunks!: number | null;
  @ApiProperty({ type: ApiFailureDto, nullable: true }) error!: ApiFailureDto | null;
}

export class DocumentDto {
  @ApiProperty({ format: 'uuid' }) id!: string;
  @ApiProperty() name!: string;
  @ApiProperty() mimeType!: string;
  @ApiProperty() sizeBytes!: number;
  @ApiProperty({ enum: ['queued', 'processing', 'ready', 'failed', 'deleted'] }) status!: string;
  @ApiProperty({ format: 'uuid', nullable: true }) activeGenerationId!: string | null;
  @ApiProperty({ type: JobDto, nullable: true }) latestJob!: JobDto | null;
  @ApiProperty({ format: 'date-time' }) createdAt!: string;
  @ApiProperty({ format: 'date-time' }) updatedAt!: string;
}

export class DocumentPageDto {
  @ApiProperty({ type: [DocumentDto] }) items!: DocumentDto[];
  @ApiProperty({ nullable: true }) nextCursor!: string | null;
}

export class SourceDto {
  @ApiProperty({ format: 'uuid' }) documentId!: string;
  @ApiProperty({ format: 'uuid' }) generationId!: string;
  @ApiProperty({ format: 'uuid' }) chunkId!: string;
  @ApiProperty() documentName!: string;
  @ApiProperty({ nullable: true }) pageStart!: number | null;
  @ApiProperty({ nullable: true }) pageEnd!: number | null;
  @ApiProperty({ nullable: true }) heading!: string | null;
  @ApiProperty() text!: string;
}
