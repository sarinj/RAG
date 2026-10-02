import { z } from 'zod';
import { InternalChatResponseSchema } from './chat.contract';

const requestId = z.string().uuid();
export const StreamProgressSchema = z.object({
  type: z.literal('progress'),
  requestId,
  stage: z.enum([
    'planner', 'retrieval', 'multi_hop', 'corpus_processing', 'reranking',
    'evidence_selection', 'evidence_sufficiency', 'generation', 'claim_validation',
    'answer_review', 'repair', 'finalization',
  ]),
  status: z.enum(['active', 'complete', 'skipped']),
  message: z.string().min(1).max(512),
}).strict();

export const InternalStreamEventSchema = z.discriminatedUnion('type', [
  StreamProgressSchema,
  z.object({ type: z.literal('answer_delta'), requestId, delta: z.string().max(32_768) }).strict(),
  z.object({ type: z.literal('answer_reset'), requestId }).strict(),
  z.object({ type: z.literal('result'), requestId, response: InternalChatResponseSchema }).strict(),
  z.object({
    type: z.literal('error'), requestId, code: z.string().max(100),
    message: z.string().max(1024), status: z.number().int().min(400).max(599),
  }).strict(),
]);

export type InternalStreamEvent = z.infer<typeof InternalStreamEventSchema>;
export type LiveAnswerEvent = Exclude<InternalStreamEvent, { type: 'result' | 'error' }>;
