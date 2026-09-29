import { Controller, Get, HttpCode, HttpStatus } from '@nestjs/common';
import { ApiOkResponse, ApiOperation, ApiResponse, ApiTags } from '@nestjs/swagger';
import { unavailable } from '../../common/errors/api-error';
import { HealthService } from './health.service';

@ApiTags('health')
@Controller('health')
export class HealthController {
  constructor(private readonly health: HealthService) {}

  @Get('live')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ summary: 'Process liveness' })
  @ApiOkResponse({ schema: { type: 'object', required: ['status'], additionalProperties: false, properties: { status: { type: 'string', enum: ['ok'] } } } })
  live(): { status: 'ok' } {
    return { status: 'ok' };
  }

  @Get('ready')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ summary: 'Database, storage, index config, and internal model readiness' })
  @ApiOkResponse({ schema: { type: 'object', required: ['status'], additionalProperties: false, properties: { status: { type: 'string', enum: ['ready'] } } } })
  @ApiResponse({ status: 503, description: 'One or more backend dependencies are not ready.' })
  async ready(): Promise<{ status: 'ready' }> {
    if (!(await this.health.isReady())) throw unavailable('MODEL_UNAVAILABLE', 'Backend dependencies are not ready.');
    return { status: 'ready' };
  }
}
