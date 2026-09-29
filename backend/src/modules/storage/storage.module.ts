import { Module } from '@nestjs/common';
import { LocalFileStorage } from './storage.service';

@Module({ providers: [LocalFileStorage], exports: [LocalFileStorage] })
export class StorageModule {}
