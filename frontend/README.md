# Sourcebook frontend

Next.js App Router UI for the RAG MVP. Use Node.js 22 and pnpm. The browser calls the public NestJS API only; it never calls the Python service or Ollama.

## Local startup

```sh
cd frontend
pnpm install
cp .env.example .env.local
pnpm dev
```

Set `NEXT_PUBLIC_API_BASE_URL` to a browser-reachable NestJS URL ending in `/api`, for example `http://localhost:3001/api`. The app shows a configuration message when the value is missing or malformed. Only public configuration belongs in this variable.

## API types and fixtures

`pnpm generate:api` generates `src/generated/api.ts` from `../backend/openapi.json`. Document and chat API models derive from those generated DTOs. The current OpenAPI document emits several nullable scalar fields and the numeric pagination limit as empty objects; the frontend narrows those fields to the v1 contract shapes in its API modules. No fixture transport or fake document, ingestion, answer, or citation data is enabled.

## Available commands

- `pnpm dev` — start the development server
- `pnpm lint` — lint the frontend
- `pnpm typecheck` — run TypeScript checking
- `pnpm test` — run focused scope, request ID, polling, and source error tests
- `pnpm generate:api` — generate client API types from `../backend/openapi.json`
- `pnpm build` — create the production build
- `pnpm start` — serve a production build

The real upload → ready → ask → citation flow still depends on the NestJS and Python services.
