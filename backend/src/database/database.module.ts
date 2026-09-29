import { Global, Logger, Module, OnApplicationShutdown, OnModuleInit } from '@nestjs/common';
import { DataSource } from 'typeorm';
import { parseServerConfig } from '../config/configuration';
import { DATA_SOURCE, SERVER_CONFIG } from '../config/tokens';
import { createDataSourceOptions } from './data-source';

class DataSourceLifecycle implements OnModuleInit, OnApplicationShutdown {
  private readonly logger = new Logger(DataSourceLifecycle.name);
  private timer?: NodeJS.Timeout;
  private connecting = false;

  constructor(private readonly dataSource: DataSource) {}

  onModuleInit(): void {
    void this.initialize();
    this.timer = setInterval(() => void this.initialize(), 5_000);
    this.timer.unref();
  }

  async onApplicationShutdown(): Promise<void> {
    if (this.timer) clearInterval(this.timer);
    if (this.dataSource.isInitialized) await this.dataSource.destroy();
  }

  private async initialize(): Promise<void> {
    if (this.dataSource.isInitialized || this.connecting) return;
    this.connecting = true;
    try {
      await this.dataSource.initialize();
      this.logger.log('PostgreSQL connection is ready.');
    } catch {
      this.logger.warn('PostgreSQL is unavailable; readiness checks will continue to fail until it recovers.');
    } finally {
      this.connecting = false;
    }
  }
}

@Global()
@Module({
  providers: [
    { provide: SERVER_CONFIG, useFactory: () => parseServerConfig() },
    {
      provide: DATA_SOURCE,
      useFactory: () => new DataSource(createDataSourceOptions()),
    },
    {
      provide: DataSourceLifecycle,
      useFactory: (dataSource: DataSource) => new DataSourceLifecycle(dataSource),
      inject: [DATA_SOURCE],
    },
  ],
  exports: [SERVER_CONFIG, DATA_SOURCE],
})
export class DatabaseModule {}
