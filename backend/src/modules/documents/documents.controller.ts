import {
  Body,
  Controller,
  Delete,
  Get,
  HttpCode,
  HttpStatus,
  Param,
  ParseUUIDPipe,
  Post,
  Query,
  UploadedFile,
  UseInterceptors,
} from '@nestjs/common';
import {
  ApiAcceptedResponse,
  ApiBody,
  ApiConsumes,
  ApiCreatedResponse,
  ApiExtraModels,
  ApiNoContentResponse,
  ApiOkResponse,
  ApiOperation,
  ApiParam,
  ApiResponse,
  ApiTags,
  getSchemaPath,
} from '@nestjs/swagger';
import { FileInterceptor } from '@nestjs/platform-express';
import { parseServerConfig } from '../../config/configuration';
import { PaginationDto } from '../../common/dto/pagination.dto';
import { EmptyDto } from './dto/empty.dto';
import { ApiFailureDto, DocumentDto, DocumentPageDto, SourceDto } from './dto/document.dto';
import { DocumentsService } from './documents.service';

const uploadLimitBytes = parseServerConfig().maxUploadBytes;

@ApiTags('documents')
@ApiExtraModels(ApiFailureDto, DocumentDto, DocumentPageDto, SourceDto)
@Controller('api/documents')
export class DocumentsController {
  constructor(private readonly documents: DocumentsService) {}

  @Post()
  @HttpCode(HttpStatus.ACCEPTED)
  @UseInterceptors(FileInterceptor('file', { limits: { fileSize: uploadLimitBytes, files: 1, fields: 0 } }))
  @ApiOperation({ summary: 'Upload a document and queue ingestion' })
  @ApiConsumes('multipart/form-data')
  @ApiBody({ schema: { type: 'object', required: ['file'], properties: { file: { type: 'string', format: 'binary' } }, additionalProperties: false } })
  @ApiAcceptedResponse({ type: DocumentDto })
  @ApiResponse({ status: 400, description: 'Invalid upload request.' })
  @ApiResponse({ status: 409, description: 'An active document with this content already exists.' })
  @ApiResponse({ status: 413, description: 'File or corpus limit exceeded.' })
  @ApiResponse({ status: 415, description: 'Unsupported file type.' })
  @ApiResponse({ status: 503, description: 'Index configuration is unavailable.' })
  upload(@UploadedFile() file: Express.Multer.File): Promise<unknown> {
    return this.documents.upload(file);
  }

  @Get()
  @ApiOperation({ summary: 'List non-deleted documents' })
  @ApiOkResponse({ type: DocumentPageDto })
  list(@Query() pagination: PaginationDto): Promise<unknown> {
    return this.documents.list(pagination.limit, pagination.cursor);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get a document, including a deleted tombstone' })
  @ApiParam({ name: 'id', format: 'uuid' })
  @ApiOkResponse({ type: DocumentDto })
  @ApiResponse({ status: 404, description: 'Document not found.' })
  get(@Param('id', ParseUUIDPipe) id: string): Promise<unknown> {
    return this.documents.get(id);
  }

  @Post(':id/retry')
  @HttpCode(HttpStatus.ACCEPTED)
  @ApiOperation({ summary: 'Retry a failed initial ingestion' })
  @ApiParam({ name: 'id', format: 'uuid' })
  @ApiBody({ type: EmptyDto })
  @ApiAcceptedResponse({ type: DocumentDto })
  @ApiResponse({ status: 409, description: 'Document is not in a retryable state.' })
  retry(@Param('id', ParseUUIDPipe) id: string, @Body() _body: EmptyDto): Promise<unknown> {
    return this.documents.retry(id);
  }

  @Post(':id/reindex')
  @HttpCode(HttpStatus.ACCEPTED)
  @ApiOperation({ summary: 'Queue a replacement index generation' })
  @ApiParam({ name: 'id', format: 'uuid' })
  @ApiBody({ type: EmptyDto })
  @ApiAcceptedResponse({ type: DocumentDto })
  @ApiResponse({ status: 409, description: 'Document is not ready to reindex.' })
  @ApiResponse({ status: 503, description: 'Index configuration is unavailable.' })
  reindex(@Param('id', ParseUUIDPipe) id: string, @Body() _body: EmptyDto): Promise<unknown> {
    return this.documents.reindex(id);
  }

  @Delete(':id')
  @HttpCode(HttpStatus.NO_CONTENT)
  @ApiOperation({ summary: 'Delete a document and schedule file/index cleanup' })
  @ApiParam({ name: 'id', format: 'uuid' })
  @ApiNoContentResponse()
  @ApiResponse({ status: 404, description: 'Document not found.' })
  async delete(@Param('id', ParseUUIDPipe) id: string): Promise<void> {
    await this.documents.delete(id);
  }

  @Get(':id/chunks/:chunkId')
  @ApiOperation({ summary: 'Read the original text and provenance for a source chunk' })
  @ApiParam({ name: 'id', format: 'uuid' })
  @ApiParam({ name: 'chunkId', format: 'uuid' })
  @ApiOkResponse({ type: SourceDto })
  @ApiResponse({ status: 404, description: 'Source not found.' })
  @ApiResponse({ status: 410, description: 'The source document has been deleted.' })
  source(
    @Param('id', ParseUUIDPipe) id: string,
    @Param('chunkId', ParseUUIDPipe) chunkId: string,
  ): Promise<Record<string, unknown>> {
    return this.documents.getSource(id, chunkId);
  }
}
