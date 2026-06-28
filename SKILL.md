# SKILL.md — L2 Assistant Project Context

## 1. Project purpose and overview

This workspace contains a ServiceNow-focused **GenAI L2 Support Assistant**. Its goal is to help Level-2 support engineers resolve incidents faster by combining:

- historical incident patterns,
- knowledge base articles,
- CMDB context,
- retrieval-augmented generation (RAG), and
- an embedded ServiceNow sidebar experience.

The intended user flow is:

1. A ServiceNow incident exists or is created.
2. The backend ingests or loads the incident.
3. The incident text is cleaned, masked for PII, classified, and chunked.
4. Similar incidents and KB chunks are retrieved from a vector store plus BM25 index.
5. An LLM generates a structured recommendation with root-cause guidance, triage steps, and a resolution draft.
6. Engineers review results in the ServiceNow sidebar, ask follow-up questions, and submit feedback.
7. Feedback updates source quality weighting over time.

The repository also includes:

- local development infrastructure (`Docker`, `docker-compose`),
- database models and async persistence,
- Celery workers and scheduled maintenance jobs,
- a ServiceNow widget and simulator,
- synthetic seed/evaluation data,
- evaluation harnesses for retrieval and generation quality.

## 2. Reality check: implementation status vs plan

The codebase is substantial and mostly scaffolded, but some parts are **fully implemented**, while others are **partially wired** or still use **placeholders/mock flows**.

### Working / implemented areas

- FastAPI application structure and routing
- Pydantic settings and environment configuration
- SQLAlchemy async models and session management
- Pinecone + pgvector abstraction layer
- Redis cache wrapper
- ServiceNow async client and mock client
- PII anonymizer
- Ticket preprocessing and KB chunking
- Hybrid retriever + reranker + context assembly + LLM client + `RAGPipeline` class
- Celery configuration and task scheduling
- ServiceNow widget assets and simulator UI
- Seed and evaluation scripts

### Partially wired or inconsistent areas

- Root `README.md` at workspace root was broken/outdated before this rewrite.
- Alembic is configured, but the `alembic/versions/` directory does not contain generated migrations.
- `scripts/bootstrap_index.py` is mostly a placeholder orchestration shell.
- `scripts/seed_test_data.py` defines synthetic KB articles but currently only inserts incidents into Postgres.
- `tests/unit/test_pii_anonymizer.py` contains an inline anonymizer implementation instead of importing the real module.
- The new `app/ingestion/pipeline.py` bridge makes the historical-incident → Postgres → embedding/vector-store path much more runnable.
- Worker and chat entrypoints now use the class-based `RAGPipeline`/retriever stack, but some docstrings/comments still describe older placeholder behavior.
- pgvector filtering remains simpler than the richer Pinecone-style filter objects produced by the retriever.

When documenting or running this project, assume it is a **strong implementation scaffold plus many real modules**, but not a completely production-wired system yet.

## 3. Tech stack and key dependencies

### Runtime and language

- Python `3.14` is the intended project runtime
- Alignment in this repo:
  - `pyproject.toml` pins Python `>=3.14,<3.15`
  - local workspace `.venv` is Python `3.14`
  - `Dockerfile` uses `python:3.14-slim`
  - Ruff/Mypy target Python `3.14`

### Backend framework

- `FastAPI`
- `Uvicorn`
- `Pydantic v2`
- `pydantic-settings`

### AI / RAG

- `openai`
- `anthropic`
- `tiktoken`
- `numpy`
- `sentence-transformers`
- `rank-bm25`

### Storage and infrastructure

- `SQLAlchemy` async
- `asyncpg`
- `PostgreSQL 15`
- `pgvector`
- `redis`
- `celery[redis]`
- `pinecone-client`

### HTTP / retries / observability

- `httpx`
- `tenacity`
- `structlog`
- `prometheus-fastapi-instrumentator`
- `prometheus-client`

### Dev/test tooling

- `pytest`
- `pytest-asyncio`
- `pytest-mock`
- `pytest-cov`
- `ruff`
- `mypy`
- `pre-commit`

## 4. Workspace structure and file-by-file explanation

## Workspace root: `D:\Projects\L2_Assistant`

- `README.md`
  - Root onboarding document for the whole workspace.
  - Rewritten to explain how to use the nested product project in `genai-l2-assistant/`.
- `SKILL.md`
  - This project-context file.
- `genai-l2-assistant-implementation-plan.md`
  - The original phased build plan.
  - Important for understanding intended architecture, but not fully aligned with the shipped code.
- `walkthrough.md`
  - Prior generated summary of the build.
  - Useful for broad orientation, but partially optimistic relative to the implementation.
- `walkthrough_2.md`
  - Additional walkthrough/reference material.
- `L2_Assistant.iml`
  - IntelliJ project metadata.
- `.git/`
  - Git metadata.
- `.idea/`
  - JetBrains IDE project settings.
- `.venv/`
  - Local virtual environment; not source of truth for dependency declarations.

## Product root: `genai-l2-assistant/`

- `.env`
  - Local environment file used by the current workspace.
- `.env.example`
  - Template of required environment variables.
- `.gitignore`
  - Ignore rules for Python artifacts, local env, etc.
- `README.md`
  - Nested project README from earlier state; no longer the main source of truth after the root rewrite.
- `pyproject.toml`
  - Poetry metadata and dependency declarations.
- `requirements.txt`
  - Pip-compatible runtime dependency list.
- `requirements-dev.txt`
  - Pip-compatible dev dependency list.
- `Dockerfile`
  - Container image for the API/worker runtime.
- `docker-compose.yml`
  - Local multi-service stack: API, Postgres, Redis, worker, beat.
- `Makefile`
  - Developer shortcuts (intended for Unix-like shells; Windows users may prefer direct commands).
- `alembic.ini`
  - Alembic configuration.
- `search_kb.py`
  - Auxiliary/search-related script entry point.
- `alembic/`
  - Alembic environment files.
  - `versions/` is present but currently empty of real migrations.
- `app/`
  - Main Python application package.
- `docs/`
  - Architecture/runbook/API docs, but not fully current.
- `infra/`
  - SQL bootstrap, GitHub Actions, Grafana dashboard, Kubernetes manifests.
- `scripts/`
  - Operational/dev scripts for seeding, bootstrapping, and evaluation.
- `servicenow/`
  - Business rule, flow export, widget files, simulator/demo pages.
- `tests/`
  - Unit, integration, and evaluation test code/data.

## `app/` package

### `app/__init__.py`

- Defines package version: `1.0.0`.

### `app/config.py`

Central settings module using `pydantic-settings`.

Defines:
- `LLMProvider`
- `VectorStoreProvider`
- `AppEnvironment`
- `LogFormat`
- `LLMSettings`
- `EmbeddingSettings`
- `VectorStoreSettings`
- `ServiceNowSettings`
- `DatabaseSettings`
- `ObservabilitySettings`
- `AppSettings`
- `get_settings()` cached singleton

Key behaviors:
- loads `.env`
- separates app/db/LLM/vector/ServiceNow concerns
- exposes `cors_origins`, `is_development`, `is_production`
- derives sync DB URL for Alembic from async URL

### `app/main.py`

FastAPI entry point.

Responsibilities:
- configure structured logging with `structlog`
- create app instance
- wire middleware
- register routers
- expose `/health` and `/metrics`
- perform startup connectivity checks for DB and Redis
- close DB engine on shutdown

### `app/api/`

#### `app/api/middleware.py`

Contains:
- `RBACMiddleware`
  - enforces `X-Engineer-Id` and `X-Engineer-Role`
  - in development, injects default headers if absent
- `RequestLoggingMiddleware`
  - logs method, path, status, latency
  - adds `X-Response-Time-Ms`
- `validate_hmac_signature()`
  - validates webhook payload signatures

#### `app/api/routes/health.py`

- exposes `GET /health`
- checks DB and Redis connectivity
- returns status, version, timestamp, and checks map

#### `app/api/routes/incidents.py`

Primary incident API.

Models:
- `AnalyzeRequest`
- `AnalyzeResponse`
- `SimilarIncidentResponse`
- `WebhookResponse`
- `IncidentListItem`

Endpoints:
- `GET /api/v1/incidents`
  - list all incidents and latest AI status
- `POST /api/v1/incidents/analyze`
  - queue or return recommendation for an incident already in DB
- `GET /api/v1/incidents/{snow_sys_id}/recommendation`
  - return cached recommendation or queue new analysis
- `GET /api/v1/incidents/{snow_sys_id}/similar`
  - returns stored similar incidents from latest recommendation record
- `POST /api/v1/incidents/{snow_sys_id}/webhook`
  - validates HMAC and enqueues analysis task

Important behavior:
- rejects resolved/closed incidents for AI analysis
- uses a 30-minute recommendation cache TTL
- requires the incident to already exist in Postgres for analyze/recommendation endpoints

#### `app/api/routes/recommendations.py`

- `GET /api/v1/recommendations/{incident_id}`
- fetches the latest recommendation using internal incident UUID, not ServiceNow sys_id

#### `app/api/routes/feedback.py`

- `POST /api/v1/feedback`
- stores engineer feedback against a recommendation

#### `app/api/routes/chat.py`

- `POST /api/v1/chat`
- supports either internal incident UUID or ServiceNow sys_id
- persists chat history in `chat_sessions`
- loads the latest stored `RecommendationDB` row for the incident before chatting
- builds a `RecommendationResult` from stored data and calls `RAGPipeline.chat()`
- returns 404 when no recommendation exists yet for the requested incident

### `app/models/`

#### `app/models/incident.py`

Defines core incident and chunk types:
- `IncidentState`
- `IncidentType`
- `ChunkType`
- `SourceType`
- `IncidentRecord`
- `KBArticle`
- `CMDBRecord`
- `IncidentQueryParams`
- `KBQueryParams`
- `ExtractedEntity`
- `ProcessedTicket`
- `TextChunk`

#### `app/models/recommendation.py`

Defines structured AI outputs:
- `TriageStep`
- `SimilarIncident`
- `KBReference`
- `SourceReference`
- `RecommendationResult`

#### `app/models/feedback.py`

Defines feedback structures:
- `FeedbackRecord`
- `FeedbackSubmission`
- `FeedbackResponse`
- `FeedbackStats`
- `FeedbackWeight`

#### `app/models/chat.py`

Defines chat and LLM interaction models:
- `ChatMessage`
- `ChatSession`
- `ChatRequest`
- `ChatResponse`
- `LLMPrompt`
- `LLMResponse`

### `app/utils/`

#### `app/utils/text_utils.py`

Utility functions:
- `strip_html`
- `normalize_whitespace`
- `clean_text`
- `split_sentences`
- `chunk_text_by_sentences`
- `count_tokens`
- `truncate_to_tokens`
- `extract_section_title`
- `combine_incident_text`

#### `app/utils/retry.py`

Retry decorators:
- `async_retry`
- `sync_retry`

### `app/governance/`

#### `app/governance/pii_anonymizer.py`

One of the more complete governance modules.

Defines:
- `PIIType`
- `PIIMatch`
- `AnonymizedResult`
- `SafetyCheckResult`
- `PIIAnonymizer`

Detects/masks:
- emails
- IPv4 and IPv6 addresses
- phone numbers
- credit cards (Luhn-validated)
- hostnames/FQDNs
- cloud keys (AWS/Azure/GCP)
- person names in ServiceNow-like context strings

Explicitly preserves UUIDs.

#### `app/governance/rbac.py`

Dedicated RBAC domain logic module separate from HTTP middleware.

Used to represent/centralize role-based access rules.

#### `app/governance/audit_logger.py`

Structured auditing helper for compliance and observability.

### `app/ingestion/`

#### `app/ingestion/servicenow_client.py`

Async ServiceNow client with:
- `httpx.AsyncClient`
- basic auth or OAuth-style token handling
- retry on rate-limit/transient errors
- typed output models

Provides:
- `get_incident()`
- `list_incidents()`
- `list_kb_articles()`
- `get_cmdb_ci()`
- `update_incident_work_note()`
- `update_incident_resolution()`

Important note:
- the OAuth implementation currently uses a **password grant** request shape, even though comments describe OAuth/client-credentials preference.

#### `app/ingestion/mock_client.py`

Drop-in mock replacement for local/test use.

Contains:
- synthetic incidents
- synthetic KB articles
- synthetic CMDB relationships
- async API compatible with `ServiceNowClient`

This is a major local-development data source.

#### `app/ingestion/ticket_processor.py`

Main incident preprocessing pipeline.

Classes:
- `TicketPreprocessor`
- `TicketChunker`

`TicketPreprocessor` does:
- combine incident fields
- clean text
- anonymize PII
- extract regex-based entities
- extract frequency-based keywords
- classify incident type
- build short summary

`TicketChunker` does:
- chunk cleaned incident text by sentence boundaries
- emit `description` chunks
- emit separate `resolution` chunks when available
- attach retrieval metadata

#### `app/ingestion/kb_processor.py`

Class:
- `KBArticleProcessor`

Does:
- split HTML/markdown-like article sections by headers
- strip HTML
- chunk section text with overlap
- emit KB `TextChunk` objects with metadata such as section title and article title

#### `app/ingestion/cmdb_enricher.py`

Class:
- `CMDBEnricher`

Provides:
- CI enrichment
- upstream/downstream dependency interpretation
- blast-radius estimation
- CMDB context formatting

#### `app/ingestion/embedding_pipeline.py`

Class:
- `EmbeddingPipeline`

Provides:
- `run_batch()`
- `run_single()`
- vector record construction
- embedding API invocation
- source deletion helper

Important note:
- this module talks directly to the OpenAI embeddings API instead of reusing `app.core.embedder.Embedder`.
- it is functional as a separate pipeline but overlaps conceptually with the core embedder abstraction.

#### `app/ingestion/pipeline.py`

High-level bridge module for historical incident persistence and vectorization.

Provides:
- `store_incident_record()`
  - upserts an `IncidentRecord` into `IncidentDB`
- `incident_db_to_record()`
  - converts a stored database row back into the domain model
- `process_and_index()`
  - preprocesses a stored incident, chunks it, embeds it through `EmbeddingPipeline`, and upserts the resulting vectors

This module is the clearest answer to “how do previous incidents get stored and vectorized?” in the current codebase.

### `app/core/`

#### `app/core/embedder.py`

Class:
- `Embedder`

Capabilities:
- OpenAI or HuggingFace embeddings
- Redis-backed caching
- token counting
- rate limiting via token bucket
- L2 normalization
- batched embedding with concurrency control

#### `app/core/retriever.py`

Key models:
- `RetrievalFilters`
- `RetrievalQuery`
- `RetrievedChunk`

Class:
- `HybridRetriever`

Capabilities:
- dense retrieval via vector store
- sparse retrieval via BM25
- reciprocal rank fusion (RRF)
- filter translation to metadata filters
- similar incident derivation

Important notes:
- BM25 index is in-memory and must be rebuilt/populated explicitly.
- dense retrieval now supports the project's async `VectorStore.query(...)` interface in addition to older sync query clients.
- pgvector filter handling in the vector store expects simple equality filters; the retriever produces Pinecone-style `$in` filters, so pgvector filtering may not behave as intended without adaptation.

#### `app/core/reranker.py`

Class:
- `Reranker`

Behavior:
- heuristic reranking by default
- optional cross-encoder loading if configured
- boosts resolution chunks, dual dense+sparse matches, source types, keyword overlap

#### `app/core/context_assembler.py`

Classes:
- `AssembledContext`
- `ContextAssembler`

Behavior:
- builds final LLM context window
- budgets around `6000` tokens
- prioritizes current incident summary, KB content, similar incidents, then extra chunks
- tracks used source IDs

#### `app/core/llm_client.py`

Classes/exceptions:
- `LLMError`
- `LLMRateLimitError`
- `LLMAuthenticationError`
- `LLMTimeoutError`
- `LLMClient`

Capabilities:
- OpenAI direct
- Azure OpenAI
- Anthropic
- retries with `tenacity`
- optional LangSmith tracing
- streaming support

#### `app/core/rag_pipeline.py`

Most important orchestration module.

Contains:
- prompt templates for recommendation/chat/resolution draft
- class `RAGPipeline`

Major methods:
- `_parse_llm_json()`
- `_build_recommendation()`
- `_store_recommendation()`
- `_audit_event()`
- `analyze_incident()`
- `_fallback_recommendation()`
- `chat()`
- `generate_resolution_draft()`

Actual orchestration path inside `analyze_incident()`:
1. build query from processed ticket
2. retrieve chunks
3. rerank chunks
4. fetch similar incidents
5. assemble context
6. call LLM
7. parse JSON output
8. create `RecommendationResult`
9. persist recommendation
10. emit audit event

Important note:
- this is implemented as a **class method flow** and is now used directly by the worker/chat integration instead of missing top-level helper functions.
- recommendation persistence and audit logging occur when a real `db_session_factory` is supplied.

#### `app/core/feedback_processor.py`

Classes:
- `SourceSignals`
- `ReviewFlag`
- `FeedbackProcessor`

Behavior:
- aggregate feedback
- compute quality scores using `positive / (positive + negative + 1)`
- load recommendation sources
- update `feedback_weights`
- queue high-confidence negative outcomes for review

### `app/storage/`

#### `app/storage/postgres.py`

Defines SQLAlchemy async models and session management.

Tables/classes:
- `IncidentDB`
- `RecommendationDB`
- `FeedbackDB`
- `AuditEventDB`
- `ChatSessionDB`
- `FeedbackWeightDB`
- `ReviewQueueDB`

Session helpers:
- `get_engine()`
- `get_session_factory()`
- `get_db_session()`
- `init_db()`
- `close_db()`

Important note:
- `init_db()` creates tables directly from metadata and is currently the most reliable local initialization path because Alembic migrations are not present.

#### `app/storage/vector_store.py`

Data models:
- `VectorRecord`
- `QueryMatch`
- `UpsertResult`
- `IndexStats`

Abstract base:
- `VectorStore`

Implementations:
- `PineconeVectorStore`
- `PGVectorStore`

Factory:
- `get_vector_store()`

Important notes:
- Pinecone path wraps sync client calls in threads.
- pgvector path creates its own `vector_embeddings` table lazily.
- pgvector query filter support is currently simplistic relative to retriever filter objects.

#### `app/storage/cache.py`

Class:
- `RedisCache`

Helpers:
- `get_cache()`
- `close_cache()`

Provides:
- `get`, `set`, `delete`, `exists`, `set_many`, `get_many`, `clear_prefix`

### `app/workers/`

#### `app/workers/celery_app.py`

Creates Celery app.

Configures:
- broker/backend from Redis
- queues/routes
- concurrency
- beat schedule

Scheduled tasks:
- `nightly_reindex` at 02:00 UTC
- `bm25_rebuild` at 03:00 UTC
- `process_feedback` at 04:00 UTC

#### `app/workers/ingestion_worker.py`

Celery tasks:
- `analyze_incident_async`
- `index_resolved_ticket`

Behavior:
- wraps async logic in sync Celery task entrypoints
- chooses `ServiceNowClient` when credentials exist and otherwise falls back to `MockServiceNowClient`
- stores or refreshes incidents in Postgres via `store_incident_record()`
- preprocesses incidents and runs `RAGPipeline.analyze_incident()` with a real session factory for persistence
- indexes resolved tickets through `app.ingestion.pipeline.process_and_index()`

#### `app/workers/reindex_worker.py`

Celery tasks:
- `nightly_reindex`
- `rebuild_bm25`
- `process_feedback_nightly`

Behavior:
- scans unindexed resolved incidents
- reuses `app.ingestion.pipeline.process_and_index()` for vector indexing
- rebuilds BM25 through `HybridRetriever.rebuild_bm25_index()` using incident text loaded from Postgres
- feedback processing updates `feedback_weights` based on recent ratings

## 5. ServiceNow assets

## `servicenow/business_rules/trigger_ai_analysis.js`

- outbound webhook trigger for incident events
- meant to call backend webhook endpoint with HMAC signature

## `servicenow/flow_designer/ai_analysis_flow.json`

- exported ServiceNow flow definition
- documents intended orchestration on the ServiceNow side

## `servicenow/widget/`

- `ai_sidebar.html`
  - main sidebar markup
- `ai_sidebar.js`
  - widget controller/client logic
- `ai_sidebar.css`
  - widget styles
- `simulator.html`
  - standalone local simulator for testing widget experience without ServiceNow
- `index.html`
  - large dashboard/demo page
- `dashboard_demo.html`
  - additional demo asset

The simulator is the easiest way to visualize the frontend locally.

## 6. Scripts and dataset sources

## Local/synthetic datasets in this repo

There is **no bundled real production ServiceNow export** in the repository.

Instead, the project uses three main local data sources:

1. `scripts/seed_test_data.py`
   - synthetic incidents for Postgres seeding
2. `app/ingestion/mock_client.py`
   - synthetic incidents, KB articles, and CMDB records for API-like local testing
3. `tests/eval/data/retrieval_eval.jsonl`
   - evaluation dataset for retrieval metrics

## `scripts/seed_test_data.py`

Purpose:
- generate synthetic local development incidents
- initialize DB tables via `init_db()`
- clear existing app data tables
- insert incident records

Important limitation:
- synthetic KB articles are defined in the file, but `insert_into_database()` currently does **not** persist them to a KB table because the Postgres schema does not have one.

## `scripts/bootstrap_index.py`

Intended purpose:
- historical bulk indexing of incidents and KB articles into vector storage

Current state:
- mostly placeholder shell functions (`fetch_resolved_incidents`, `fetch_kb_articles`, `process_and_chunk`, `index_chunks`) that currently log and return empty lists.

## `scripts/eval_run.py`

Purpose:
- run retrieval or generation evaluations
- output JSON report
- enforce quality gates

Quality gates:
- retrieval: `MRR >= 0.5`
- generation: all score dimensions must pass configured thresholds

## Evaluation dataset: `tests/eval/data/retrieval_eval.jsonl`

Contains 30+ JSONL rows with fields like:
- `query_id`
- `query`
- `relevant_doc_ids`
- `category`
- `metadata`

Coverage spans categories such as:
- application
- infrastructure
- network
- access management

## 7. Tests

## Unit tests

- `tests/unit/test_pii_anonymizer.py`
  - broad pattern coverage, but uses inline test implementation rather than importing production code
- `tests/unit/test_rag_pipeline.py`
  - pipeline-oriented behavior checks
- `tests/unit/test_retriever.py`
  - retrieval and ranking checks
- `tests/unit/test_ticket_processor.py`
  - preprocessing/chunking checks

## Integration tests

- `tests/integration/test_servicenow_client.py`
- `tests/integration/test_vector_store.py`

These validate important boundaries but may require environment-aware setup.

## Evaluation tests

- `tests/eval/eval_retrieval.py`
- `tests/eval/eval_generation.py`

These are closer to benchmark harnesses than standard unit tests.

## 8. How the project runs end to end

## Intended full architecture

1. ServiceNow incident triggers webhook/business rule.
2. FastAPI receives webhook.
3. Celery analysis task runs.
4. Incident is loaded or fetched.
5. Preprocessing masks PII, extracts features, classifies type.
6. Retrieval uses vector store + BM25.
7. Context is assembled.
8. LLM generates recommendation JSON.
9. Recommendation is stored.
10. Sidebar polls and displays results.
11. Engineer gives feedback.
12. Nightly jobs reindex and adjust feedback weights.

## Most reliable current local path

Given the current code wiring, the most reliable local path is:

1. Start Postgres + Redis (+ optional worker/beat) with Docker Compose.
2. Initialize tables using `scripts/seed_test_data.py` or `init_db()`.
3. Seed synthetic incidents.
4. Start the API with Uvicorn.
5. Open `/docs`, `/health`, and optionally `servicenow/widget/simulator.html`.
6. Use endpoints that operate on existing seeded incidents.
7. Trigger analysis before chat so a stored recommendation exists, and use the worker/indexing paths for more realistic background execution.

## 9. Environment setup and configuration

Important environment variables:

### LLM
- `LLM_PROVIDER`
- `LLM_MODEL_NAME`
- `OPENAI_API_KEY`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`
- `ANTHROPIC_API_KEY`

### Embeddings
- `EMBEDDING_MODEL`
- `EMBEDDING_DIMENSIONS`

### Vector store
- `VECTOR_STORE_PROVIDER`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `PINECONE_ENVIRONMENT`

### ServiceNow
- `SNOW_INSTANCE_URL`
- `SNOW_USERNAME`
- `SNOW_PASSWORD`
- `SNOW_CLIENT_ID`
- `SNOW_CLIENT_SECRET`
- `SNOW_WEBHOOK_SECRET`

### Database/cache
- `DATABASE_URL`
- `REDIS_URL`

### App behavior
- `APP_ENV`
- `LOG_FORMAT`
- `SECRET_KEY`
- `ALLOWED_ORIGINS`

### Observability
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT`
- `PROMETHEUS_ENABLED`

## 10. Important design patterns and conventions

### Pattern: Pydantic everywhere at boundaries

The project consistently uses Pydantic models for:
- inbound API payloads
- ServiceNow records
- processed incident structures
- recommendation outputs
- feedback/chat payloads

### Pattern: async-first backend design

- FastAPI routes are async
- SQLAlchemy uses async engine/session
- ServiceNow client is async
- Redis client is async
- Celery workers wrap async code in sync task entrypoints

### Pattern: separation by responsibility

- `api/` = HTTP layer
- `core/` = AI/RAG logic
- `ingestion/` = data acquisition and preprocessing
- `storage/` = persistence and vector backends
- `governance/` = security/compliance concerns
- `workers/` = background execution
- `models/` = domain/data contracts

### Pattern: retrieval metadata enrichment

Chunks carry metadata such as:
- `source_id`
- `source_type`
- `category`
- `subcategory`
- `cmdb_ci`
- `assignment_group`
- timestamps
- keywords

This metadata is used for filtering, source attribution, and context formatting.

### Pattern: graceful fallback behavior

When some advanced modules are unavailable, the code often logs and returns placeholder responses rather than crashing.

This is useful for scaffolding, but it can hide incomplete wiring if you expect full production-grade AI behavior.

## 11. Common errors and gotchas specific to this repo

### 1. Root vs nested project confusion

The actual Python app lives in `genai-l2-assistant/`, not at the workspace root.

### 2. README drift

Older markdown files describe the intended architecture, not always the exact current implementation.

### 3. Python version alignment

- active project metadata is pinned to 3.14
- implementation plan now says Python 3.14
- CI runs on Python 3.14
- Dockerfile and local `.venv` use 3.14

Best practical choice for this repo as-is: **Python 3.14**.

### 4. Alembic is configured but migrations are missing

`alembic upgrade head` alone is not enough to create the application tables unless migrations are added.

For local setup today, use:
- `scripts/seed_test_data.py`, or
- `app.storage.postgres.init_db()`

### 5. Seed script only inserts incidents

Although synthetic KB articles are declared, they are not stored in Postgres by the current seed path.

### 6. Chat depends on an existing recommendation

`app/api/routes/chat.py` now uses `RAGPipeline.chat()`, but it expects the incident to already have a stored `RecommendationDB` row.

If no recommendation exists yet, the route returns 404 and you should analyze the incident first.

### 7. Worker/bootstrap maturity is uneven

`app/workers/ingestion_worker.py` and `app/workers/reindex_worker.py` are now much more runnable, but `scripts/bootstrap_index.py` is still not the most reliable entrypoint for full historical indexing.

### 8. Reindex/BM25 worker now uses in-repo implementations

`app/workers/reindex_worker.py` now uses `app.ingestion.pipeline.process_and_index()` and `HybridRetriever.rebuild_bm25_index()`.

The remaining limitation is not missing modules; it is that BM25 state is still rebuilt in memory and provider-specific vector filtering still differs between Pinecone and pgvector.

### 9. `Makefile` may be less convenient on Windows

The workspace is on Windows and uses PowerShell by default. Direct Python/Docker commands are often easier than `make` unless GNU Make is installed.

### 10. Vector-store provider expectations differ

Retriever filter output is Pinecone-style (`$in`, `$gte`), while pgvector query code expects much simpler key/value filtering.

### 11. Integration with real ServiceNow/OpenAI requires env completeness

You need valid values for:
- DB
- Redis
- ServiceNow auth
- LLM provider auth
- vector store provider auth

Otherwise some flows start but degrade into placeholder behavior or fail at runtime.

### 12. Unit tests do not always validate production implementations directly

At least one key test file (`test_pii_anonymizer.py`) uses an inline anonymizer implementation rather than importing the production module.

## 12. Recommended mental model for new contributors

Think of the repository as four layers:

1. **Platform layer**
   - FastAPI, settings, logging, Docker, Celery, Postgres, Redis
2. **Data layer**
   - ServiceNow ingestion, ticket cleaning, KB chunking, CMDB enrichment
3. **AI/RAG layer**
   - embeddings, retrieval, reranking, context assembly, LLM generation
4. **Experience layer**
   - REST API, ServiceNow widget, simulator, feedback loop

If you need to debug a feature, trace it in this order:

- environment/config → route → worker/pipeline → storage/retrieval → frontend/simulator

## 13. Best next improvements if continuing development

Highest-impact fixes would be:

1. Expose and wire real top-level worker entrypoints to the `RAGPipeline` class.
2. Add real Alembic migrations.
3. Create a persisted KB/article storage model if KB content must be tracked in Postgres.
4. Align retriever filters with pgvector filtering semantics.
5. Replace placeholder bootstrap logic with real ServiceNow + embedding pipeline orchestration.
6. Update tests to import and validate the production anonymizer and other production modules directly.
7. Decide and document one canonical Python version across plan, Docker, linting, and local setup.

## 14. Short operational summary

If someone asks “what is this project, really?” the short answer is:

> It is a Python/FastAPI + Celery + Postgres/Redis + Pinecone/pgvector RAG assistant for ServiceNow L2 incident resolution, with solid core modules and a polished frontend simulator, but with some worker/bootstrap wiring still incomplete and several setup details that need to be handled carefully for local execution.



