import 'dotenv/config';
import 'reflect-metadata';
import { ValidationPipe } from '@nestjs/common';
import { NestFactory } from '@nestjs/core';
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger';
import { Request, Response, NextFunction } from 'express';
import { AppModule } from './app.module';
import { HttpExceptionFilter } from './common/errors/http-exception.filter';
import { RequestIdMiddleware } from './common/middleware/request-id.middleware';
import { parseServerConfig } from './config/configuration';
import { addContractErrors } from './openapi/contract-errors';

async function bootstrap(): Promise<void> {
  const config = parseServerConfig();
  const app = await NestFactory.create(AppModule, { bufferLogs: true });
  app.enableShutdownHooks();
  const requestIdMiddleware = new RequestIdMiddleware();
  app.use((request: Request, response: Response, next: NextFunction) => requestIdMiddleware.use(request, response, next));
  app.enableCors({
    origin: config.frontendOrigin,
    methods: ['GET', 'POST', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'X-Request-Id'],
    exposedHeaders: ['X-Request-Id'],
  });
  app.useGlobalPipes(new ValidationPipe({
    transform: true,
    whitelist: true,
    forbidNonWhitelisted: true,
    forbidUnknownValues: true,
  }));
  app.useGlobalFilters(new HttpExceptionFilter());

  const swaggerConfig = new DocumentBuilder()
    .setTitle('Learning RAG Product API')
    .setDescription('Public API for document ingestion, saved conversations, and cited answers.')
    .setVersion('1.0.0')
    .build();
  const openapi = addContractErrors(SwaggerModule.createDocument(app, swaggerConfig));
  SwaggerModule.setup('/docs', app, openapi, { jsonDocumentUrl: '/openapi.json' });

  await app.listen(config.port, '0.0.0.0');
}

void bootstrap();
