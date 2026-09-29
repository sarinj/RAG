import 'reflect-metadata';
import { writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { NestFactory } from '@nestjs/core';
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger';
import { OpenApiModule } from './openapi.module';
import { addContractErrors } from './openapi/contract-errors';

async function main(): Promise<void> {
  const app = await NestFactory.create(OpenApiModule, { logger: false });
  const options = new DocumentBuilder()
    .setTitle('Learning RAG Product API')
    .setDescription('Public API for document ingestion, saved conversations, and cited answers.')
    .setVersion('1.0.0')
    .build();
  const document = addContractErrors(SwaggerModule.createDocument(app, options));
  await writeFile(resolve(process.cwd(), 'openapi.json'), `${JSON.stringify(document, null, 2)}\n`, 'utf8');
  await app.close();
  process.stdout.write('Wrote backend/openapi.json (OpenAPI 1.0.0).\n');
}

main().catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.message : 'OpenAPI export failed.'}\n`);
  process.exitCode = 1;
});
