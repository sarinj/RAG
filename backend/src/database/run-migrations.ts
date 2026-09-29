import 'dotenv/config';
import { AppDataSource } from './data-source';

async function main(): Promise<void> {
  await AppDataSource.initialize();
  try {
    const migrations = await AppDataSource.runMigrations({ transaction: 'all' });
    process.stdout.write(`Applied ${migrations.length} migration(s).\n`);
  } finally {
    await AppDataSource.destroy();
  }
}

main().catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.message : 'Migration failed.'}\n`);
  process.exitCode = 1;
});
