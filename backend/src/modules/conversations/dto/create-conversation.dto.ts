import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { IsObject, IsOptional, IsString } from 'class-validator';

export class CreateConversationDto {
  @ApiPropertyOptional({ minLength: 1, maxLength: 120, example: 'Research notes' })
  @IsOptional()
  @IsString()
  title?: string;

  @ApiProperty({
    oneOf: [
      { type: 'object', required: ['mode'], properties: { mode: { type: 'string', enum: ['all'] } }, additionalProperties: false },
      { type: 'object', required: ['mode', 'documentIds'], properties: { mode: { type: 'string', enum: ['selected'] }, documentIds: { type: 'array', items: { type: 'string', format: 'uuid' }, minItems: 1, maxItems: 100 } }, additionalProperties: false },
    ],
  })
  @IsObject()
  scope!: unknown;
}
