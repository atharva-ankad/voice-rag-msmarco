# Project State: Voice-Enabled Multilingual RAG

## Current Phase: Phase 8 (BM25 In-Memory / SQLite Lexical Index) & Phase 6 (Embeddings Preparation)
**Objective:** Build the local BM25 index over the chunks for zero-latency lexical retrieval, and configure credentials for Qdrant Cloud & Embedding API.

## Completed Phases
* [x] **Phase 1 & 2:** Dataset ingestion and canonical SQLite schema.
  * Queries ingested: 1,000
  * Passages flattened: 9,988
* [x] **Phase 4 & 5:** Chunking engine and metadata storage.
  * Total chunks generated: 76,037
  * Strategies: `semantic_300`, `fixed_256_50`, `fixed_512_100`
  * Languages: English (`en`) and Hindi (`hi`)

## Locked-in Schemas (SQLite)
* **Table `passages`**: `passage_id` (PK), `query_id`, `source_language`, `target_language`, `english_text`, `translated_text`, `is_selected`, `split`
* **Table `chunks`**: `chunk_id` (PK), `passage_id` (FK), `strategy`, `text`, `token_count`