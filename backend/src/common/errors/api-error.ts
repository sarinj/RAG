import { HttpException } from '@nestjs/common';

export class ApiError extends HttpException {
  constructor(
    readonly code: string,
    message: string,
    status: number,
  ) {
    super({ code, message }, status);
  }
}

export function badRequest(message = 'The request is invalid.'): ApiError {
  return new ApiError('VALIDATION_ERROR', message, 400);
}

export function notFound(message = 'The requested resource was not found.'): ApiError {
  return new ApiError('NOT_FOUND', message, 404);
}

export function conflict(code: string, message: string): ApiError {
  return new ApiError(code, message, 409);
}

export function unavailable(code = 'MODEL_UNAVAILABLE', message = 'The required service is unavailable.'): ApiError {
  return new ApiError(code, message, 503);
}
