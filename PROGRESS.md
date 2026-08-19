# Project State: Voice-Enabled Multilingual RAG

## Completed Phases
* **Phase 1 & 2:** Dataset ingestion and canonical SQLite schema. 
  * *Dataset Split:* `validation` split, English (`en`) and Hindi (`hi`).
  * *Records:* 1,000 valid queries, 9,988 flattened passages.
  * *Constraint:* Dropped queries where `sum(is_selected) == 0`.
* **Phase 4 & 5:** Chunking engine and metadata storage.
  * *Records:* 76,037 total chunks generated.
  * *Strategies Executed:* `semantic_300` (26,438), `fixed_256_50` (28,510), `fixed_512_100` (21,089).
* **Phase 8:** BM25 Lexical Index.
  * *Status:* Pickled dictionary `{'chunk_ids': [...], 'bm25': bm25_obj}` into `cache/bm25_semantic_300.pkl`.
* **Phase 6 & 7:** Vector Embedding & Qdrant Indexing.
  * *Status:* Processed completely on Google Colab (Tesla T4). Generated 1024-dimensional dense vectors using `BAAI/bge-m3` in FP16 precision.
  * *Upload:* Upserted all 76,037 points to Qdrant Cloud collection `msmarco_multilingual_bge`. IDs are deterministically hashed via `uuid.uuid5()`.
  * *Indexing:* Created a keyword payload index for the `strategy` field.
* **Phase 9:** Hybrid Search Retrieval.
  * *Status:* Implemented Reciprocal Rank Fusion (RRF) using $k = 60$.
  * *Architecture:* Combines BM25 lexical token-matching with Qdrant dense vector search (using `query_points` API).

## Bottleneck / Current Finding
Pure hybrid search (BM25 + bge-m3 bi-encoder) fails on causal dependencies and antonyms (e.g., retrieving passages about "low blood pressure" or "high blood pressure causes stroke" for the query "What causes high blood pressure?"). 

## Next Immediate Objective
**Phase 9.5 (Cross-Encoder Re-ranking):** 
Integrate a cross-encoder model (e.g., `BAAI/bge-reranker-base`) on top of the hybrid retrieval output to fix semantic blindness and strictly enforce syntactic/causal relationships.