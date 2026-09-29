import { z } from "zod";

const apiUrlSchema = z
  .string()
  .url("Use an absolute browser-reachable URL, such as http://localhost:3001/api")
  .refine((value) => {
    const url = new URL(value);
    return (url.protocol === "http:" || url.protocol === "https:")
      && !url.username
      && !url.password
      && !url.search
      && !url.hash
      && url.pathname.replace(/\/$/, "").endsWith("/api");
  }, "The public API URL must use http(s) and end with /api");

const rawApiUrl = process.env.NEXT_PUBLIC_API_BASE_URL?.trim() ?? "";
const parsedApiUrl = rawApiUrl ? apiUrlSchema.safeParse(rawApiUrl) : null;

export const publicConfig = {
  apiBaseUrl: parsedApiUrl?.success ? parsedApiUrl.data.replace(/\/$/, "") : null,
  apiConfigError: !rawApiUrl
    ? "Set NEXT_PUBLIC_API_BASE_URL in frontend/.env.local to connect the public NestJS API."
    : parsedApiUrl && !parsedApiUrl.success
      ? parsedApiUrl.error.issues[0]?.message ?? "NEXT_PUBLIC_API_BASE_URL is invalid."
      : null,
};

export function isApiConfigured() {
  return publicConfig.apiBaseUrl !== null;
}
