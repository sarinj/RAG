import { AlertCircle, LoaderCircle, Settings2 } from "lucide-react";
import { publicConfig } from "@/config/env";
import { apiErrorMessage } from "@/lib/http";
import { Button } from "./button";

export function ApiConfigNotice() {
  return (
    <section className="api-config-notice" role="status">
      <div className="feedback-icon"><Settings2 size={18} aria-hidden="true" /></div>
      <div>
        <h2>Connect the public API</h2>
        <p>{publicConfig.apiConfigError}</p>
        <p className="config-command"><code>cp .env.example .env.local</code> in <code>frontend/</code></p>
      </div>
    </section>
  );
}

export function ErrorNotice({ error, onRetry, title = "Something went wrong" }: {
  error: unknown;
  onRetry?: () => void;
  title?: string;
}) {
  return (
    <div className="error-notice" role="alert">
      <AlertCircle size={18} aria-hidden="true" />
      <div className="error-notice-content">
        <strong>{title}</strong>
        <p>{apiErrorMessage(error)}</p>
        {onRetry && <Button variant="outline" size="sm" onClick={onRetry}>Try again</Button>}
      </div>
    </div>
  );
}

export function LoadingState({ label = "Loading" }: { label?: string }) {
  return (
    <div className="loading-state" role="status" aria-live="polite">
      <LoaderCircle size={18} className="spin" aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}
