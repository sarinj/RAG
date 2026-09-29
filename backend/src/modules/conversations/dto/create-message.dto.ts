import { ApiProperty } from '@nestjs/swagger';
import { IsString, IsUUID } from 'class-validator';

export class CreateMessageDto {
  @ApiProperty({ format: 'uuid' })
  @IsUUID()
  requestId!: string;

  @ApiProperty({ minLength: 1, maxLength: 8000 })
  @IsString()
  content!: string;
}
