import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { z } from 'zod';
import { ServerConfig } from './configuration';

export const IndexingArtifactSchema = z.object({
  embeddingModel: z.string().min(1),
  embeddingDigest: z.string().min(1),
  embeddingDimension: z.number().int().min(1).max(16000),
  chunkTargetTokens: z.number().int().positive(),
  chunkOverlapTokens: z.number().int().nonnegative(),
  chunkerVersion: z.string().min(1),
  tokenizerVersion: z.string().min(1),
  contextPromptVersion: z.string().min(1).nullable(),
  contextModel: z.string().min(1).nullable(),
  contextModelDigest: z.string().min(1).nullable(),
  contextWindowTokens: z.number().int().positive(),
  maxOutputTokens: z.number().int().positive(),
  retrievalMode: z.enum(['vector', 'hybrid']).default('hybrid'),
}).passthrough().superRefine((artifact, context) => {
  if (artifact.chunkOverlapTokens >= artifact.chunkTargetTokens) {
    context.addIssue({ code: 'custom', message: 'chunk overlap must be less than chunk target' });
  }
});

export type IndexingArtifact = z.infer<typeof IndexingArtifactSchema>;

export function readIndexingArtifact(config: ServerConfig): IndexingArtifact {
  let raw: unknown;
  try {
    raw = JSON.parse(readFileSync(resolve(config.indexConfigPath), 'utf8'));
  } catch {
    throw new Error('The shared indexing configuration is unavailable.');
  }

  const result = IndexingArtifactSchema.safeParse(raw);
  if (!result.success) {
    throw new Error('The shared indexing configuration is invalid.');
  }
  if (config.indexMode === 'contextual' && (!result.data.contextPromptVersion || !result.data.contextModel || !result.data.contextModelDigest)) {
    throw new Error('Contextual indexing requires a complete context model configuration.');
  }
  return result.data;
}
