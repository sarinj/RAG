import { MigrationInterface, QueryRunner } from 'typeorm';

export class DirectAnswerStatus1710000002000 implements MigrationInterface {
  name = 'DirectAnswerStatus1710000002000';

  async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query('ALTER TABLE messages DROP CONSTRAINT IF EXISTS messages_answer_status_check');
    await queryRunner.query(`
      UPDATE messages
      SET answer_status = 'direct_answer'
      WHERE answer_status = 'small_talk'
    `);
    await queryRunner.query(`
      ALTER TABLE messages
      ADD CONSTRAINT messages_answer_status_check
      CHECK (answer_status IS NULL OR answer_status IN ('answered','insufficient_evidence','direct_answer'))
    `);
  }

  async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query('ALTER TABLE messages DROP CONSTRAINT IF EXISTS messages_answer_status_check');
    await queryRunner.query(`
      UPDATE messages
      SET answer_status = 'insufficient_evidence'
      WHERE answer_status = 'direct_answer'
    `);
    await queryRunner.query(`
      ALTER TABLE messages
      ADD CONSTRAINT messages_answer_status_check
      CHECK (answer_status IS NULL OR answer_status IN ('answered','insufficient_evidence','small_talk'))
    `);
  }
}
