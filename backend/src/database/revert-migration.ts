import 'dotenv/config';
import { AppDataSource } from './data-source';

async function main(): Promise<void> {
  await AppDataSource.initialize();
  try {
    await AppDataSource.undoLastMigration({ transaction: 'all' });
    process.stdout.write('Reverted the latest migration.\n');
  } finally {
    await AppDataSource.destroy();
  }
}

main().catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.message : 'Migration revert failed.'}\n`);
  process.exitCode = 1;
});
