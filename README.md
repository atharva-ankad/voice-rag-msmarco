<div align="center">

# 🎙️ Multilingual Voice-Enabled RAG System (MS MARCO)

### Enterprise-grade, low-latency Retrieval-Augmented Generation with real-time speech I/O for **English (`en-IN`)** and **Hindi (`hi-IN`)**

[Voice RAG — Live Demo](https://voice-rag-msmarco-eight.vercel.app/)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-ASGI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Modal](https://img.shields.io/badge/Modal-Serverless%20GPU-7C3AED)](https://modal.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-DC244C)](https://qdrant.tech/)
[![Vercel](https://img.shields.io/badge/Vercel-Edge%20CDN-000000?logo=vercel&logoColor=white)](https://vercel.com/)
[![Sarvam AI](https://img.shields.io/badge/Sarvam%20AI-105B%20%7C%20Saaras%20v3-FF6B35)](https://www.sarvam.ai/)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20T4-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com/)
[![Guardrails](https://img.shields.io/badge/Guardrails-3--Tier%20Defense-critical)](#-3-tier-hybrid-defense-system-deep-dive)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Voice in → Hybrid Retrieval → Reranking → Guarded Generation**
</div>

---

## 🚀 Executive Summary & Key Features

This project is a **production-grade conversational question-answering system** built over the MS MARCO validation corpus, engineered to answer spoken or typed questions in **English and Hindi** with strictly corpus-grounded responses.

Rather than relying on a single-shot vector lookup, the system runs a **multi-stage retrieval funnel** - dense semantic search and lexical BM25 executed in parallel, fused via Reciprocal Rank Fusion, then narrowed by a cross-encoder reranker - before any tokens reach the LLM. A **3-tier guardrail stack** wraps the pipeline end-to-end: adversarial input is rejected before it costs a single GPU cycle, low-confidence retrievals are refused rather than hallucinated over, and every generated answer is cross-examined against its own source passages before it is returned to the user.

### Key Features

| | Feature | Detail |
|:--:|---|---|
| 🎤 | **Full-duplex voice interface** | Sarvam Saaras `saaras:v3` STT with automatic `en-IN`/`hi-IN` detection; Sarvam Neural TTS returns Base64 audio for instant playback |
| 🌐 | **True multilingual parity** | Parallel English + translated Hindi MS MARCO splits, language-tagged at the chunk level |
| 🔀 | **Hybrid multi-stage retrieval** | Dense `BAAI/bge-m3` (1024-d, cosine) ⊕ Okapi BM25 → RRF (`k=60`) → cross-encoder rerank → top-5 context |
| 🎯 | **FP16 cross-encoder reranking** | `BAAI/bge-reranker-v2-m3` on GPU, converting recall-oriented candidates into precision-oriented context |
| 🛡️ | **3-tier hybrid defense** | NemoGuard-8B injection/toxicity shield → relevance gate (`τ ≥ 0.35`) → Sarvam-105B factuality validator |
| 🧩 | **Decoupled storage architecture** | Qdrant Cloud holds only vectors + light payload; SQLite sidecar holds full chunk text |
| ✂️ | **Pluggable chunking strategies** | Semantic windowing (default) plus two fixed-window variants |
| ⚡ | **Serverless GPU backend** | Modal Labs container on a dedicated NVIDIA T4, weights pre-baked into image layers |
| 📊 | **Transparent retrieval** | Every response ships source `chunk_id`, `doc_id`, RRF score, reranker confidence, and latency breakdown |

---

## 🏗️ System Architecture

### End-to-End Flow

![](https://github.com/atharva-ankad/voice-rag-msmarco/blob/main/docs/flowchart_rag_final.png)

### Retrieval Funnel (ASCII)

```
                                  User Query (EN / HI)
                                           │
                      ┌────────────────────┴────────────────────┐
                      ▼                                         ▼
        ┌─────────────────────────────┐           ┌─────────────────────────────┐
        │   DENSE - Qdrant Cloud      │           │   LEXICAL - Okapi BM25      │
        │   BAAI/bge-m3 · 1024-d      │           │   cache/bm25_semantic_300   │
        │   COSINE · ANN search       │           │   exact token frequency     │
        └──────────────┬──────────────┘           └──────────────┬──────────────┘
                       │  Top 50                                 │  Top 50
                       └────────────────────┬────────────────────┘
                                            ▼
                        ┌───────────────────────────────────────┐
                        │   RECIPROCAL RANK FUSION  (k = 60)    │
                        │   score = Σ 1 / (k + rank_i(d))       │
                        └───────────────────┬───────────────────┘
                                            │  Top 30 candidates
                                            ▼
                        ┌───────────────────────────────────────┐
                        │   CROSS-ENCODER RERANKER (FP16 · GPU) │
                        │   BAAI/bge-reranker-v2-m3             │
                        │   sequence classification (Q, P)      │
                        └───────────────────┬───────────────────┘
                                            │  Top 5 context chunks
                                            ▼
                             ┌──────────────────────────┐
                             │  Relevance Gate τ ≥ 0.35 │
                             └──────────────┬───────────┘
                                            ▼
                                  Sarvam-105B Generation
```

### Voice Pipeline

```
Frontend Audio (.wav / .mp3 / webm - MediaRecorder or file upload)
              │
              ▼
   Sarvam Saaras:v3 STT  ──►  Transcribed Text (auto-detected en-IN / hi-IN)
              │
              ▼
   [ Guardrails → Hybrid RAG → Reranker → Sarvam-105B ]
              │
              ▼
   Text Response
```

---

## 🧰 Complete Tech Stack

| Layer / Domain | Technology / Library | Specification / Configuration |
|---|---|---|
| **Data & Storage** | MS MARCO Validation Set | Multilingual - English + parallel translated Hindi validation splits |
| | SQLite (`rag_sidecar.db`) | Document sidecar: chunk text payload + metadata, decoupled from the vector index |
| | Qdrant Cloud | Vector database - collection `msmarco_bge_m3`, `COSINE` metric, 1024-dim dense vectors |
| | `parquet_metadata.bin` | Raw dataset serialized index offsets for fast source lookup |
| **Embeddings & Search** | `BAAI/bge-m3` | 1024-dimensional normalized dense semantic representation (FP16 on T4) |
| | `rank-bm25` (Okapi BM25) | Exact token matching & lexical frequency scoring, serialized to `cache/bm25_semantic_300.pkl` |
| | Reciprocal Rank Fusion | Scale-free rank aggregator, constant `k = 60`, fuses dense + sparse rankings |
| | `BAAI/bge-reranker-v2-m3` | Cross-encoder sequence classification, FP16 precision, GPU-resident |
| **Generation & Defense** | Sarvam AI `sarvam-105B` | Primary LLM inference engine + Layer-3 hallucination verification (temp `0.2`, max tokens `512`) |
| | NVIDIA NeMo Guardrails (`NemoGuard-8B`) | Layer 1 - content safety, toxicity & prompt-injection shield |
| | Local Cosine Evaluator | Layer 2 - retrieval relevance score filtering (`τ ≥ 0.35`) |
| **Voice Processing** | Sarvam Saaras `saaras:v3` | Multilingual audio transcription / STT with automatic language detection |
| **Backend & Cloud** | FastAPI + Uvicorn | High-throughput asynchronous ASGI web server, CORS-enabled |
| | Pydantic | Strict request/response schema validation (`config/schemas.py`) |
| | Modal Labs | Serverless GPU container runtime - NVIDIA T4, `debian_slim` (Python 3.11) |
| **Frontend** | Vanilla JS / HTML5 / CSS3 | MediaRecorder mic capture, WAV/MP3 upload, audio player, transcript & source viewer |
| | Vercel | Production edge CDN hosting |

---

## 📂 Project Directory Tree

```
voice-rag-msmarco/
├── app.py                    # FastAPI endpoints, CORS, orchestrator
├── modal_deploy.py           # Serverless GPU deployment
├── requirements.txt
│
├── cache/                    # Serialized BM25 index
├── chunking/                 # Chunking strategies
├── config/                   # Pydantic schemas
├── frontend/                 # UI (HTML/JS/CSS)
├── generation/                # LLM invocation + 3-tier guardrails
├── ingestion/                 # DB init, data load, chunking, embedding
├── interfaces/                 # Voice (STT/TTS) handler
├── retrieval/                  # BM25, cross-encoder, hybrid search
└── tests/                      # E2E, guardrail & latency tests
```

---

## 📥 Data Ingestion & Indexing Pipeline

The ingestion pipeline is a strict five-stage sequence. Each stage is idempotent and safe to re-run, but stages **must** execute in order - every downstream stage consumes the artifact produced by the previous one.

```
load_data.py  →  init_db.py  →  populate_chunks.py  →  embed_and_upload.py  →  bm25_index.py
   (parse)        (schema)        (chunk + store)        (dense index)          (lexical index)
```

| Strategy | Window | Overlap | When to use |
|---|:--:|:--:|---|
| **A - Semantic Windowing** *(default)* | 300 tokens | 1 sentence | Balanced context retention and semantic granularity; preserves boundary meaning across passage transitions |
| **B1 - Fixed Window Small** | 256 tokens | 50 tokens | Maximum retrieval precision for short, fact-dense queries |
| **B2 - Fixed Window Large** | 512 tokens | 100 tokens | Broader context for multi-hop or explanatory questions |

## 🛡️ 3-Tier Hybrid Defense System

Guardrails are positioned at three distinct points in the request lifecycle, each targeting a failure mode the others structurally cannot catch. The ordering is deliberate: the cheapest, highest-recall filter runs first so that adversarial traffic never reaches the GPU.

```
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  QUERY                                                                   │
   │    │                                                                     │
   │    ▼                                                                     │
   │  ╔═══════════════════════════════════════════════════════════╗           │
   │  ║ L1 · CONTENT SAFETY & INJECTION SHIELD  (NemoGuard-8B)    ║ PRE-RETRIEVAL
   │  ║ Blocks: prompt injection · jailbreaks · toxicity · PII    ║           │
   │  ╚═══════════════════════════════════════════════════════════╝           │
   │    │ pass                                                                │
   │    ▼   [ Hybrid Retrieval → RRF → Cross-Encoder ]                        │
   │  ╔═══════════════════════════════════════════════════════════╗           │
   │  ║ L2 · CONTEXT RELEVANCE GATE  (local cosine scorer)        ║ PRE-GENERATION
   │  ║ Blocks: out-of-corpus questions · weak-evidence context   ║           │
   │  ╚═══════════════════════════════════════════════════════════╝           │
   │    │ pass                                                                │
   │    ▼   [ Sarvam-105B Generation ]                                        │
   │  ╔═══════════════════════════════════════════════════════════╗           │
   │  ║ L3 · HALLUCINATION & FACTUALITY VALIDATOR  (Sarvam-105B)  ║ POST-GENERATION
   │  ║ Blocks: ungrounded claims · unsupported extrapolation     ║           │
   │  ╚═══════════════════════════════════════════════════════════╝           │
   │    │ pass                                                                │
   │    ▼  VERIFIED ANSWER → TTS                                              │
   └──────────────────────────────────────────────────────────────────────────┘
```

### Defense Matrix

| | Layer 1 | Layer 2 | Layer 3 |
|---|:--:|:--:|:--:|
| **Stage** | Pre-retrieval | Pre-generation | Post-generation |
| **Engine** | NemoGuard-8B | Local cosine scorer | Sarvam-105B |
| **Threat** | Adversarial input | Irrelevant context | Ungrounded output |
| **Cost when triggered** | Near-zero | Retrieval only | Full generation |

---

## 🔌 API Reference

Base URL (production): `https://<workspace>--voice-rag-msmarco-fastapi-app.modal.run`

| Method | Endpoint | Content-Type | Purpose |
|---|---|---|---|
| `POST` | `/query` | `application/json` | Text-in → text-out RAG query |
| `POST` | `/voice-query` | `multipart/form-data` | Audio-in → text + Base64 audio-out RAG query |

---

## 📄 License & Acknowledgments

**Built with:**

- [MS MARCO](https://microsoft.github.io/msmarco/) - Microsoft Machine Reading Comprehension dataset
- [BAAI FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) - `bge-m3` embeddings and `bge-reranker-v2-m3`
- [Sarvam AI](https://www.sarvam.ai/) - `sarvam-105B` LLM, Saaras `v3` STT, Neural TTS
- [NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) - programmable LLM safety rails
- [Qdrant](https://qdrant.tech/) - high-performance vector database
- [Modal Labs](https://modal.com/) - serverless GPU infrastructure
- [FastAPI](https://fastapi.tiangolo.com/) · [rank-bm25](https://github.com/dorianbrown/rank_bm25) · [Vercel](https://vercel.com/)

<div align="center">

---

**Built for Corpus-grounded, multilingual question answering.**

</div>
