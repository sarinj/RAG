import { randomUUID } from 'node:crypto';
import { Injectable, NestMiddleware } from '@nestjs/common';
import { NextFunction, Request, Response } from 'express';

type RequestWithId = Request & { requestId?: string };
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

@Injectable()
export class RequestIdMiddleware implements NestMiddleware {
  use(request: RequestWithId, response: Response, next: NextFunction): void {
    const supplied = request.header('x-request-id');
    const requestId = supplied && UUID.test(supplied) ? supplied : randomUUID();
    request.requestId = requestId;
    response.setHeader('X-Request-Id', requestId);
    next();
  }
}
