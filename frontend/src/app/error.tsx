"use client";

import { AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function AppError({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <section className="route-error">
      <span className="route-error-icon"><AlertTriangle size={22} aria-hidden="true" /></span>
      <h1>This page hit a snag</h1>
      <p>Your saved documents and conversations are safe. Try loading this page again.</p>
      <Button onClick={reset}>Reload page</Button>
    </section>
  );
}
