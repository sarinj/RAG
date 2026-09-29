# Learning RAG with Anthropic’s Contextual Retrieval

This guide synthesizes Anthropic’s [Contextual Retrieval article](https://www.anthropic.com/engineering/contextual-retrieval) and its [Claude Cookbook implementation](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide). It focuses on the practical lesson: improve what each indexed chunk says about itself, and combine semantic and lexical retrieval when each catches different queries.

Research checked September 29, 2026. Provider APIs and model limits change; recheck the linked documentation when implementing.

## 1. The basic RAG loop

Retrieval-augmented generation (RAG) selects relevant material from a corpus and gives it to a language model as evidence for answering a query. A minimal pipeline is:

1. Parse documents and split them into chunks.
2. Embed each chunk and index its vector.
3. Embed a user query and find similar chunks.
4. Put the selected chunks in the model’s prompt and generate an answer.

Vector similarity finds semantically related text, but can miss exact identifiers, product names, or phrases. BM25 keyword search is good at exact terms, but less capable with paraphrases. Anthropic recommends combining both and merging their ranked results; the Cookbook’s measured basic baseline starts with vector search alone, then adds hybrid retrieval. ([Article](https://www.anthropic.com/engineering/contextual-retrieval), [Cookbook](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide))

## 2. What contextual retrieval changes

Chunking can separate a passage from the information needed to understand it. A chunk that says “revenue grew 3%” is hard to match to a question about a particular company and quarter if those names occur elsewhere in the document.

Contextual retrieval asks a language model to write a short, chunk-specific description using both the full source document and the chunk. Prepend that description to the chunk before creating its vector and BM25 index. Keep the original chunk separately for display and answer generation. ([Article](https://www.anthropic.com/engineering/contextual-retrieval), [Cookbook](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide))

The Anthropic article’s example contextualizer prompt is:

```text
<document>
{{WHOLE_DOCUMENT}}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{{CHUNK_CONTENT}}
</chunk>
Please give a short succinct context to situate this chunk within the overall
document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
```

The generated context is typically 50–100 tokens. For example:

```text
Context: This chunk is from ACME Corp's Q2 2023 SEC filing. It describes
quarter-over-quarter revenue growth; the previous quarter's revenue was $314M.

The company's revenue grew by 3% over the previous quarter.
```

Treat the context as search metadata that improves matching. Preserve provenance and send the original text (with enough surrounding context to interpret it) to the answer model.

## 3. An implementation path

### A. Prepare and chunk the corpus

Store stable identifiers and source metadata for every document and chunk: document ID, chunk ID, original text, source URI or path, heading/page, and version or updated-at time. Choose chunk boundaries that preserve useful structure (for example, sections or code blocks) and measure a few chunk sizes and overlaps rather than assuming one setting is best. Anthropic calls out chunk size, boundaries, and overlap as retrieval variables. ([Article](https://www.anthropic.com/engineering/contextual-retrieval))

The Cookbook starts from chunks grouped under source documents. Its demonstration uses pre-chunked codebases; its minimal baseline describes heading-based chunks, embeddings, and cosine similarity. It stores source IDs, chunk IDs, original content, and contextualized content as metadata. ([Cookbook](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide))

### B. Generate chunk context once at ingestion

For each document, pass its full text and one chunk at a time to the contextualizer. Process a document’s chunks together so the full document can be reused. Store the generated text and usage/cost information. The cookbook uses Anthropic’s Messages API and ephemeral prompt caching for the document text, then uses the returned context plus the original chunk as the text to embed. Context generation happens during ingestion, so normal queries do not incur this generation step. ([Cookbook](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide))

Provider-neutral sketch:

```python
for document in documents:
    chunks = split(document)
    for chunk in chunks:
        context = contextualize(document.text, chunk.text)
        searchable_text = f"{context}\n\n{chunk.text}"

        vector_index.upsert(
            id=chunk.id,
            vector=embed(searchable_text),
            metadata={
                "document_id": document.id,
                "source": document.source,
                "original_text": chunk.text,
                "context": context,
            },
        )
        keyword_index.upsert(
            id=chunk.id,
            text=f"{context}\n\n{chunk.text}",
            metadata={"document_id": document.id, "source": document.source},
        )
```

The Cookbook’s concrete demo uses Claude to contextualize, Voyage embeddings in batches of 128, an in-memory NumPy cosine-similarity vector store with pickle persistence, and Elasticsearch BM25. Those choices make a readable tutorial, not a requirement: its own text recommends replacing the in-memory store with a hosted vector database for production. ([Cookbook](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide))

### C. Retrieve broadly, fuse, and optionally rerank

For each query:

1. Search the vector index for semantic matches.
2. Search BM25 over both original and contextualized chunk text.
3. Merge/deduplicate the ranked lists. The Cookbook uses weighted reciprocal-rank-style fusion, starting with 80% semantic and 20% BM25, and retrieves up to 150 candidates from each before returning top results. Its example score is weighted `1/rank`; standard RRF adds a constant to the rank denominator. Tune the fusion method, candidate count, and weights on your own evaluation set. ([Cookbook](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide))
4. Optionally rerank the candidate chunks against the query with a cross-encoder or reranking API. Retrieve a broad candidate pool, score candidates with the query, and keep the top few. The article’s experiment retrieved 150 and reranked to 20. Reranking adds a runtime call, cost, and latency. ([Article](https://www.anthropic.com/engineering/contextual-retrieval))
5. Give the answer model the query and selected original chunks, including source identifiers. Ask it to answer from the supplied evidence and cite the source IDs. This keeps the text used for factual answers traceable to the corpus.

An RRF variant combines rank positions without comparing scores from unrelated systems:

```text
rrf_score(chunk) = Σ weight(source) / (constant + rank_in_source(chunk))
```

## 4. Evaluate before choosing complexity

Build a small, representative evaluation set of questions with the relevant chunk(s) marked. Compare each retrieval version on the same queries. Track at least:

- **Recall@k:** What fraction of all labeled relevant chunks appeared in the top *k* results? If each query has one required chunk, HitRate@k or Pass@k is a simpler yes/no measure.
- **Answer quality and grounding:** Does the generated answer correctly use retrieved evidence and point to its source?
- **Latency and cost:** Measure ingestion expense separately from per-query expense.

The Cookbook’s Pass@k metric checks whether the labeled “golden chunk” appears in the top *k*. Its test set contains 248 queries over nine codebases. The page reports Pass@10 of 87.15% for its baseline, 92.34% with contextual embeddings, 93.21% with hybrid search, and 95.26% with reranking. These are results for that particular dataset, chunking, models, and evaluation setup—not a guarantee for another corpus. ([Cookbook](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide))

The article reports a separate cross-domain experiment measured as retrieval failure at top 20: 5.7% baseline, 3.7% with contextual embeddings (35% fewer failures), 2.9% with contextual embeddings plus contextual BM25 (49% fewer), and 1.9% when reranking was also added (67% fewer). Its evaluation used its top-performing embedding configuration and tested code, fiction, and research papers. Do not combine these numbers with the Cookbook’s Pass@10 results; they use different test sets and metrics. ([Article](https://www.anthropic.com/engineering/contextual-retrieval))

## 5. Tradeoffs and cautions

- **Ingestion cost:** Context generation and embedding cost more than indexing raw chunks. Context generation is usually a one-time cost per document version; prompt caching can reduce repeated full-document input costs. When a source document changes, regenerate affected chunks’ context and vectors so the index does not mix versions. ([Article](https://www.anthropic.com/engineering/contextual-retrieval), [Cookbook](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide))
- **Runtime cost and latency:** Hybrid retrieval adds a lexical search system. Reranking adds a model/API call after initial search. Measure whether its recall/answer-quality improvement justifies the extra latency and cost. ([Article](https://www.anthropic.com/engineering/contextual-retrieval))
- **Context quality:** Generated context can be incomplete or mistaken. Use a short, domain-appropriate prompt, preserve the original text and metadata, and evaluate retrieval and answer faithfulness. Anthropic suggests tailoring the prompt to a domain, such as adding a glossary for terms defined elsewhere. ([Article](https://www.anthropic.com/engineering/contextual-retrieval))
- **Chunk count in the prompt:** More retrieved chunks may improve recall but can distract the answer model. Test values such as 5, 10, and 20 against your task instead of assuming more is always better. ([Article](https://www.anthropic.com/engineering/contextual-retrieval))
- **Small corpora:** The article’s roughly 200,000-token threshold reflects the models available in 2024. Current Claude models have different context limits, with some supporting up to 1 million tokens. Compare a cached full-context prompt with RAG using the context limit and cost of the model you actually choose. ([Article](https://www.anthropic.com/engineering/contextual-retrieval), [current model overview](https://platform.claude.com/docs/en/models/overview), [prompt caching docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching))
- **Tutorial code ages:** The Cookbook demonstrates a particular set of API clients, model names, caching configuration, and storage choices. For example, its code uses an older prompt-caching beta header; current Claude docs say prompt caching no longer requires the beta prefix. Treat the notebook as an algorithm walkthrough and check current provider API/model documentation and your chosen index’s requirements before using snippets in production. ([Cookbook](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide), [current prompt caching docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching))

## 6. What to use now

There is no single newer method that is best for every corpus. A practical current starting point for ordinary document question answering is:

1. Split by document structure and keep source, version, access, and section metadata on each chunk.
2. Optionally add chunk-specific context when chunks lose important document-level meaning.
3. Retrieve with both dense vectors and BM25, apply metadata/access filters, and merge the ranked lists with reciprocal-rank fusion.
4. Retrieve a broad candidate set (for example, around 50), rerank it against the query, and send only the best few original chunks to the answer model. Tune the numbers against your own latency and relevance targets. This hybrid-plus-reranking pattern is also documented in current retrieval architecture guidance. ([Microsoft Azure AI Search RAG retrieval guide](https://learn.microsoft.com/azure/architecture/ai-ml/guide/rag/rag-information-retrieval))

Two alternatives are useful for narrower cases:

- **Late chunking:** A long-context embedding model first encodes the whole document into token vectors; the system then pools those vectors into chunk embeddings. This gives chunks document context without generating a separate LLM-written description for each chunk. It requires an embedding model that exposes the needed token-level vectors and supports long inputs. A 2026 chunking preprint finds that contextualized chunking helps some in-corpus retrieval tasks but can hurt in-document needle retrieval, so compare it with ordinary and contextual retrieval on the questions you expect. ([Late Chunking paper](https://arxiv.org/abs/2409.04701), [2026 chunking evaluation preprint](https://arxiv.org/abs/2602.16974))
- **Graph or hierarchical retrieval:** Consider this when users ask corpus-wide questions such as “What themes appear across these reports?” rather than locating a specific fact. Microsoft GraphRAG’s global search summarizes over generated community reports and is described as resource-intensive. Microsoft marks the GraphRAG repository as a research project in maintenance mode, so treat it as a specialized pattern to evaluate rather than the default starting point. ([GraphRAG query overview](https://microsoft.github.io/graphrag/query/overview/), [Microsoft GraphRAG repository](https://github.com/microsoft/graphrag))

For a Claude application, you can pass retrieved passages as search-result content blocks with source titles and stable source IDs; the current API can attach citations to answers. Keep the original chunk as citable evidence, and treat retrieved document text as untrusted content that must not override your system instructions. ([Search results and citations](https://platform.claude.com/docs/en/build-with-claude/search-results), [prompt-injection guidance](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/mitigate-jailbreaks))

## Suggested learning exercise

Start with ordinary vector search on a small corpus and a dozen representative questions. Record the relevant chunk for each question. Add chunk-specific context and compare Recall@5/10. Then add BM25 and rank fusion, and finally try reranking. Change one layer at a time and record both retrieval results and latency/cost; keep the simplest pipeline that meets the target for your own task.
