import Link from "next/link";
import { FileQuestion } from "lucide-react";

export default function NotFound() {
  return (
    <section className="route-error">
      <span className="route-error-icon"><FileQuestion size={22} aria-hidden="true" /></span>
      <h1>We couldn’t find that page</h1>
      <p>The conversation or document may have been removed.</p>
      <Link className="button button-primary button-md" href="/documents">Go to your library</Link>
    </section>
  );
}
