import {
  Body,
  Controller,
  Get,
  HttpStatus,
  Param,
  ParseUUIDPipe,
  Post,
  Query,
  Req,
  Res,
} from '@nestjs/common';
import {
  ApiAcceptedResponse,
  ApiBody,
  ApiCreatedResponse,
  ApiExtraModels,
  ApiOkResponse,
  ApiOperation,
  ApiParam,
  ApiProduces,
  ApiResponse,
  ApiTags,
} from '@nestjs/swagger';
import { Request, Response } from 'express';
import { PaginationDto } from '../../common/dto/pagination.dto';
import { CreateConversationDto } from './dto/create-conversation.dto';
import { CreateMessageDto } from './dto/create-message.dto';
import { ConversationDto, ConversationPageDto, MessageDto, MessagePageDto, TurnDto } from './dto/conversation.dto';
import { ConversationsService } from './conversations.service';
import { ApiError, unavailable } from '../../common/errors/api-error';

type RequestWithId = Request & { requestId?: string };

@ApiTags('conversations')
@ApiExtraModels(ConversationDto, ConversationPageDto, MessageDto, MessagePageDto, TurnDto)
@Controller('api/conversations')
export class ConversationsController {
  constructor(private readonly conversations: ConversationsService) {}

  @Post()
  @ApiOperation({ summary: 'Create a conversation with a fixed document scope' })
  @ApiBody({ type: CreateConversationDto })
  @ApiCreatedResponse({ type: ConversationDto })
  @ApiResponse({ status: 400, description: 'Invalid scope or title.' })
  @ApiResponse({ status: 404, description: 'A selected document was not found.' })
  @ApiResponse({ status: 409, description: 'A selected document is not ready.' })
  create(@Body() body: CreateConversationDto): Promise<Record<string, unknown>> {
    return this.conversations.create(body);
  }

  @Get()
  @ApiOperation({ summary: 'List conversations, including empty conversations' })
  @ApiOkResponse({ type: ConversationPageDto })
  list(@Query() pagination: PaginationDto): Promise<unknown> {
    return this.conversations.list(pagination.limit, pagination.cursor);
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get a conversation and its fixed scope' })
  @ApiParam({ name: 'id', format: 'uuid' })
  @ApiOkResponse({ type: ConversationDto })
  @ApiResponse({ status: 404, description: 'Conversation not found.' })
  get(@Param('id', ParseUUIDPipe) id: string): Promise<Record<string, unknown>> {
    return this.conversations.get(id);
  }

  @Get(':id/messages')
  @ApiOperation({ summary: 'List conversation messages in chronological order' })
  @ApiParam({ name: 'id', format: 'uuid' })
  @ApiOkResponse({ type: MessagePageDto })
  @ApiResponse({ status: 404, description: 'Conversation not found.' })
  messages(
    @Param('id', ParseUUIDPipe) id: string,
    @Query() pagination: PaginationDto,
  ): Promise<unknown> {
    return this.conversations.messages(id, pagination.limit, pagination.cursor);
  }

  @Post(':id/messages')
  @ApiOperation({ summary: 'Create an idempotent chat turn' })
  @ApiParam({ name: 'id', format: 'uuid' })
  @ApiBody({ type: CreateMessageDto })
  @ApiOkResponse({ type: TurnDto, description: 'Turn completed or a previous failed turn was replayed.' })
  @ApiAcceptedResponse({ type: TurnDto, description: 'The matching request is still pending.' })
  @ApiResponse({ status: 400, description: 'Invalid question or request ID.' })
  @ApiResponse({ status: 404, description: 'Conversation not found.' })
  @ApiResponse({ status: 409, description: 'Request conflict, pending turn, invalid scope, or index mismatch.' })
  @ApiResponse({ status: 422, description: 'Context limit or citation validation failed.' })
  @ApiResponse({ status: 503, description: 'Model service unavailable.' })
  @ApiResponse({ status: 504, description: 'Answer generation timed out.' })
  async sendMessage(
    @Param('id', ParseUUIDPipe) id: string,
    @Body() body: CreateMessageDto,
    @Req() request: RequestWithId,
    @Res({ passthrough: true }) response: Response,
  ): Promise<unknown> {
    const result = await this.conversations.sendMessage(id, body.requestId, body.content, request.requestId ?? '00000000-0000-4000-8000-000000000000');
    response.status(result.httpStatus === 202 ? HttpStatus.ACCEPTED : HttpStatus.OK);
    return result.turn;
  }

  @Post(':id/messages/stream')
  @ApiOperation({
    summary: 'Stream an idempotent chat turn with public progress and provisional answer text',
    description: 'SSE data frames have types accepted, progress, answer_reset, answer_delta, done or error. '
      + 'Done contains the saved turn; a duplicate request can replay a pending or failed turn. Completed answers have passed source checks. '
      + 'Progress describes source work, not private model reasoning. '
      + 'Disconnected clients reconcile through saved history; an accepted turn keeps running within its deadline.',
  })
  @ApiParam({ name: 'id', format: 'uuid' })
  @ApiBody({ type: CreateMessageDto })
  @ApiProduces('text/event-stream')
  @ApiResponse({ status: 200, description: 'SSE stream; done follows persistence, error is terminal.', schema: { type: 'string' } })
  @ApiResponse({ status: 400, description: 'Invalid question or request ID.' })
  @ApiResponse({ status: 404, description: 'Conversation not found.' })
  @ApiResponse({ status: 409, description: 'Request conflict, pending turn, invalid scope, or index mismatch.' })
  async streamMessage(
    @Param('id', ParseUUIDPipe) id: string,
    @Body() body: CreateMessageDto,
    @Req() request: RequestWithId,
    @Res() response: Response,
  ): Promise<void> {
    let heartbeat: NodeJS.Timeout | undefined;
    const write = async (event: unknown) => {
      if (response.destroyed || response.writableEnded) return;
      if (!response.headersSent) {
        response.status(HttpStatus.OK);
        response.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
        response.setHeader('Cache-Control', 'no-cache, no-transform');
        response.setHeader('X-Accel-Buffering', 'no');
        response.flushHeaders();
        heartbeat = setInterval(() => {
          if (!response.destroyed && !response.writableEnded && !response.writableNeedDrain) response.write(': keepalive\n\n');
        }, 10_000);
        heartbeat.unref();
      }
      if (!response.write(`data: ${JSON.stringify(event)}\n\n`)) {
        if (response.destroyed || response.writableEnded) return;
        await new Promise<void>((resolve) => {
          // A slow/disconnected viewer must not hold the durable turn open indefinitely.
          const timer = setTimeout(() => {
            response.destroy();
            finish();
          }, 10_000);
          timer.unref();
          const finish = () => {
            clearTimeout(timer);
            response.off('drain', finish);
            response.off('close', finish);
            resolve();
          };
          response.once('drain', finish);
          response.once('close', finish);
        });
      }
    };
    const close = () => { if (heartbeat) clearInterval(heartbeat); };
    response.once('close', close);
    try {
      const result = await this.conversations.sendMessage(
        id, body.requestId, body.content,
        request.requestId ?? '00000000-0000-4000-8000-000000000000', write,
      );
      await write({ type: 'done', requestId: body.requestId, turn: result.turn });
    } catch (error) {
      if (!response.headersSent && !response.destroyed) throw error;
      const failure = error instanceof ApiError ? error : unavailable();
      await write({ type: 'error', requestId: body.requestId, code: failure.code, message: failure.message });
    } finally {
      close();
      response.off('close', close);
      if (response.headersSent && !response.destroyed && !response.writableEnded) response.end();
    }
  }
}
