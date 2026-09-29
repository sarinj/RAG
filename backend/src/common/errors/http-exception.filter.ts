import {
  ArgumentsHost,
  Catch,
  ExceptionFilter,
  HttpException,
  HttpStatus,
  Logger,
} from '@nestjs/common';
import { Request, Response } from 'express';
import { ApiError } from './api-error';

type RequestWithId = Request & { requestId?: string };

@Catch()
export class HttpExceptionFilter implements ExceptionFilter {
  private readonly logger = new Logger(HttpExceptionFilter.name);

  catch(exception: unknown, host: ArgumentsHost): void {
    const http = host.switchToHttp();
    const request = http.getRequest<RequestWithId>();
    const response = http.getResponse<Response>();
    const requestId = request.requestId ?? '00000000-0000-4000-8000-000000000000';

    let status = HttpStatus.INTERNAL_SERVER_ERROR;
    let code = 'INTERNAL_ERROR';
    let message = 'An unexpected error occurred.';

    if (exception instanceof ApiError) {
      status = exception.getStatus();
      code = exception.code;
      message = exception.message;
    } else if (exception instanceof HttpException) {
      status = exception.getStatus();
      const responseBody = exception.getResponse();
      if (status === 413 || (typeof responseBody === 'object' && responseBody !== null && 'code' in responseBody && responseBody.code === 'LIMIT_FILE_SIZE')) {
        status = 413;
        code = 'FILE_TOO_LARGE';
        message = 'The uploaded file exceeds the configured size limit.';
      } else if (status === 415) {
        code = 'UNSUPPORTED_FILE_TYPE';
        message = 'Only text PDFs, UTF-8 TXT, and Markdown files are supported.';
      } else if (status === 400) {
        code = 'VALIDATION_ERROR';
        message = 'The request is invalid.';
      } else if (status === 404) {
        code = 'NOT_FOUND';
        message = 'The requested resource was not found.';
      } else if (status >= 500) {
        code = 'INTERNAL_ERROR';
        message = 'An unexpected error occurred.';
      }
    } else {
      this.logger.error(`Unhandled request failure (${requestId})`, exception instanceof Error ? exception.stack : undefined);
    }

    response.status(status).json({ code, message, requestId });
  }
}
