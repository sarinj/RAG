import type { Metadata } from "next";
import { DocumentLibrary } from "@/features/documents";

export const metadata: Metadata = { title: "Documents" };

export default function DocumentsPage() {
  return <DocumentLibrary />;
}
