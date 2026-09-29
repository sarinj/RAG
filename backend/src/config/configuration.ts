import { z } from 'zod';

export const CHUNK_CAPACITY_LIMIT = 5000;

const EnvironmentSchema = z.object({
  PORT: z.coerce.number().int().min(1).max(65535).default(3001),
  CORS_ORIGIN: z.string().url().default('http://localhost:3000'),
  DATABASE_URL: z.string().min(1).default('postgres://rag:rag@localhost:5432/rag'),
  UPLOAD_DIR: z.string().min(1).default('./data/uploads'),
  MAX_UPLOAD_BYTES: z.coerce.number().int().positive().max(50 * 1024 * 1024).default(50 * 1024 * 1024),
  MAX_FILES: z.coerce.number().int().positive().default(100),
  INDEX_CONFIG_PATH: z.string().min(1).default('./config/index-config.json'),
  INDEX_MODE: z.enum(['baseline', 'contextual']).default('baseline'),
  RETRIEVAL_MODE: z.enum(['vector', 'hybrid']).default('vector'),
  LLM_BASE_URL: z.string().url().default('http://localhost:8000'),
  INTERNAL_API_TOKEN: z.string().optional(),
  INTERNAL_TIMEOUT_MS: z.coerce.number().int().positive().default(130_000),
  TURN_EXPIRY_SECONDS: z.coerce.number().int().positive().default(180),
}).superRefine((value, context) => {
  if (value.INTERNAL_TIMEOUT_MS >= value.TURN_EXPIRY_SECONDS * 1000) {
    context.addIssue({ code: 'custom', path: ['INTERNAL_TIMEOUT_MS'], message: 'Internal timeout must be shorter than turn expiry.' });
  }
});

export type ServerConfig = {
  port: number;
  frontendOrigin: string;
  databaseUrl: string;
  uploadDir: string;
  maxUploadBytes: number;
  maxFiles: number;
  indexConfigPath: string;
  indexMode: 'baseline' | 'contextual';
  retrievalMode: 'vector' | 'hybrid';
  llmBaseUrl: string;
  internalApiToken?: string;
  internalTimeoutMs: number;
  turnExpirySeconds: number;
};

export function parseServerConfig(env: NodeJS.ProcessEnv = process.env): ServerConfig {
  const parsed = EnvironmentSchema.safeParse(env);
  if (!parsed.success) {
    const fields = parsed.error.issues.map((issue) => issue.path.join('.')).join(', ');
    throw new Error(`Invalid backend environment configuration: ${fields}`);
  }

  const value = parsed.data;
  return {
    port: value.PORT,
    frontendOrigin: value.CORS_ORIGIN,
    databaseUrl: value.DATABASE_URL,
    uploadDir: value.UPLOAD_DIR,
    maxUploadBytes: value.MAX_UPLOAD_BYTES,
    maxFiles: value.MAX_FILES,
    indexConfigPath: value.INDEX_CONFIG_PATH,
    indexMode: value.INDEX_MODE,
    retrievalMode: value.RETRIEVAL_MODE,
    llmBaseUrl: value.LLM_BASE_URL.replace(/\/$/, ''),
    internalApiToken: value.INTERNAL_API_TOKEN,
    internalTimeoutMs: value.INTERNAL_TIMEOUT_MS,
    turnExpirySeconds: value.TURN_EXPIRY_SECONDS,
  };
}
