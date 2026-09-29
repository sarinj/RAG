import { defineConfig } from '@hey-api/openapi-ts';

export default defineConfig({
  input: '../llm-backend/openapi.json',
  output: { path: 'src/modules/llm-client/generated' },
  plugins: ['@hey-api/typescript'],
});
