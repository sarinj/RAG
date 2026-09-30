import { z } from 'zod';
import type { ChatRequest, ChatResponse, Citation } from '../generated';

export const InternalCitationSchema = z.object({
  label: z.string().regex(/^S[1-9][0-9]*$/),
  documentId: z.string().uuid(),
  generationId: z.string().uuid(),
  chunkId: z.string().uuid(),
  documentName: z.string().min(1),
  pageStart: z.number().int().min(1).nullable(),
  pageEnd: z.number().int().min(1).nullable(),
  heading: z.string().nullable(),
}).strict().refine(
  (citation) => citation.pageStart === null || citation.pageEnd === null || citation.pageStart <= citation.pageEnd,
  { message: 'pageStart must not be greater than pageEnd', path: ['pageEnd'] },
);

export const InternalChatResponseSchema = z.object({
  contractVersion: z.literal('1.1.0'),
  requestId: z.string().uuid(),
  answer: z.string(),
  answerStatus: z.enum(['answered', 'insufficient_evidence', 'direct_answer']),
  citations: z.array(InternalCitationSchema),
}).strict();

export type InternalCitation = Citation;
export type InternalChatResponse = ChatResponse;
export type InternalChatRequest = Omit<ChatRequest, 'contractVersion'> & { contractVersion: '1.1.0' };
