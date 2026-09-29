import { randomUUID } from 'node:crypto';
import { mkdir, open, readdir, rename, rm, stat } from 'node:fs/promises';
import { join, resolve, sep } from 'node:path';
import { Inject, Injectable, Logger, OnApplicationShutdown, OnModuleInit } from '@nestjs/common';
import { ServerConfig } from '../../config/configuration';
import { SERVER_CONFIG } from '../../config/tokens';
import { badRequest } from '../../common/errors/api-error';

export type StoredUpload = { storageKey: string };

@Injectable()
export class LocalFileStorage implements OnModuleInit, OnApplicationShutdown {
  private readonly root: string;
  private readonly logger = new Logger(LocalFileStorage.name);
  private timer?: NodeJS.Timeout;
  private preparing = false;

  constructor(@Inject(SERVER_CONFIG) private readonly config: ServerConfig) {
    this.root = resolve(config.uploadDir);
  }

  onModuleInit(): void {
    void this.prepare();
    this.timer = setInterval(() => void this.prepare(), 30_000);
    this.timer.unref();
  }

  onApplicationShutdown(): void {
    if (this.timer) clearInterval(this.timer);
  }

  async store(buffer: Buffer, extension: '.pdf' | '.txt' | '.md'): Promise<StoredUpload> {
    await mkdir(this.root, { recursive: true, mode: 0o700 });
    const id = randomUUID();
    const storageKey = `${id}${extension}`;
    const temporaryPath = join(this.root, `.upload-${id}.tmp`);
    const finalPath = this.resolveStorageKey(storageKey);
    const handle = await open(temporaryPath, 'wx', 0o600);
    try {
      await handle.writeFile(buffer);
      await handle.sync();
    } catch (error) {
      await handle.close();
      await rm(temporaryPath, { force: true });
      throw error;
    }
    await handle.close();
    try {
      await rename(temporaryPath, finalPath);
    } catch (error) {
      await rm(temporaryPath, { force: true });
      throw error;
    }
    return { storageKey };
  }

  async remove(storageKey: string): Promise<void> {
    await rm(this.resolveStorageKey(storageKey), { force: true });
  }

  async staleStorageKeys(minimumAgeMs: number): Promise<string[]> {
    const cutoff = Date.now() - minimumAgeMs;
    const candidates: string[] = [];
    for (const entry of await readdir(this.root).catch(() => [])) {
      if (!/^[0-9a-f-]{36}\.(pdf|txt|md)$/i.test(entry)) continue;
      try {
        const details = await stat(this.resolveStorageKey(entry));
        if (details.mtimeMs < cutoff) candidates.push(entry);
      } catch {
        // Concurrent deletion is safe; the next sweep will retry any surviving file.
      }
    }
    return candidates;
  }

  async isWritable(): Promise<boolean> {
    try {
      await mkdir(this.root, { recursive: true, mode: 0o700 });
      const probe = join(this.root, `.health-${randomUUID()}`);
      const handle = await open(probe, 'wx', 0o600);
      await handle.close();
      await rm(probe, { force: true });
      return true;
    } catch {
      return false;
    }
  }

  private resolveStorageKey(storageKey: string): string {
    if (!/^[0-9a-f-]{36}\.(pdf|txt|md)$/i.test(storageKey)) {
      throw badRequest('The stored file key is invalid.');
    }
    const resolved = resolve(this.root, storageKey);
    if (!resolved.startsWith(`${this.root}${sep}`)) throw badRequest('The stored file key is invalid.');
    return resolved;
  }

  private async removeStaleTemporaryFiles(): Promise<void> {
    const cutoff = Date.now() - 60 * 60 * 1000;
    let entries: string[];
    try {
      entries = await readdir(this.root);
    } catch {
      return;
    }
    await Promise.all(entries.filter((entry) => /^\.upload-[0-9a-f-]{36}\.tmp$/i.test(entry)).map(async (entry) => {
      const path = join(this.root, entry);
      try {
        const details = await stat(path);
        if (details.mtimeMs < cutoff) await rm(path, { force: true });
      } catch {
        // A concurrent cleanup or filesystem error is retried on the next startup.
      }
    }));
  }

  private async prepare(): Promise<void> {
    if (this.preparing) return;
    this.preparing = true;
    try {
      await mkdir(this.root, { recursive: true, mode: 0o700 });
      await this.removeStaleTemporaryFiles();
      if (this.timer) clearInterval(this.timer);
    } catch (error) {
      this.logger.warn(`Upload storage is unavailable; readiness checks will continue to fail until it recovers: ${error instanceof Error ? error.message : 'filesystem error'}`);
    } finally {
      this.preparing = false;
    }
  }
}
