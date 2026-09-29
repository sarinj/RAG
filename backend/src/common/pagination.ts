import { badRequest } from './errors/api-error';

export type CursorPosition = { createdAt: string; id: string };

export function encodeCursor(position: CursorPosition): string {
  return Buffer.from(JSON.stringify(position), 'utf8').toString('base64url');
}

export function decodeCursor(cursor?: string): CursorPosition | undefined {
  if (!cursor) return undefined;
  try {
    const decoded: unknown = JSON.parse(Buffer.from(cursor, 'base64url').toString('utf8'));
    if (typeof decoded !== 'object' || decoded === null || !('createdAt' in decoded) || !('id' in decoded)) throw new Error();
    const value = decoded as { createdAt: unknown; id: unknown };
    if (typeof value.createdAt !== 'string' || Number.isNaN(Date.parse(value.createdAt))) throw new Error();
    if (typeof value.id !== 'string' || !UUID.test(value.id)) throw new Error();
    return { createdAt: new Date(value.createdAt).toISOString(), id: value.id };
  } catch {
    throw badRequest('The pagination cursor is invalid.');
  }
}

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export function pageResult<T extends { createdAt: string | Date; id: string }>(rows: T[], limit: number): { rows: T[]; nextCursor: string | null } {
  const hasNext = rows.length > limit;
  const items = hasNext ? rows.slice(0, limit) : rows;
  const last = items.at(-1);
  return {
    rows: items,
    nextCursor: hasNext && last ? encodeCursor({ createdAt: new Date(last.createdAt).toISOString(), id: last.id }) : null,
  };
}
