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
  ApiResponse,
  ApiTags,
} from '@nestjs/swagger';
import { Request, Response } from 'express';
import { PaginationDto } from '../../common/dto/pagination.dto';
import { CreateConversationDto } from './dto/create-conversation.dto';
import { CreateMessageDto } from './dto/create-message.dto';
import { ConversationDto, ConversationPageDto, MessageDto, MessagePageDto, TurnDto } from './dto/conversation.dto';
import { ConversationsService } from './conversations.service';

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
}
