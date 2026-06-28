# GenAI L2 Support Assistant — Antigravity Implementation Plan

> **Purpose:** This document is a step-by-step implementation guide intended for use with Antigravity. Each phase, module, and task is written as actionable prompts and specifications that can be handed directly to Antigravity for execution. Follow phases sequentially; later phases depend on earlier ones.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository & Project Structure](#2-repository--project-structure)
3. [Phase 1 — Foundation & Environment Setup](#3-phase-1--foundation--environment-setup)
4. [Phase 2 — Data Pipeline & Indexing](#4-phase-2--data-pipeline--indexing)
5. [Phase 3 — Core AI Engine (RAG)](#5-phase-3--core-ai-engine-rag)
6. [Phase 4 — ServiceNow Integration & API Layer](#6-phase-4--servicenow-integration--api-layer)
7. [Phase 5 — Frontend UI (ServiceNow Widget)](#7-phase-5--frontend-ui-servicenow-widget)
8. [Phase 6 — Feedback Loop & Continuous Learning](#8-phase-6--feedback-loop--continuous-learning)
9. [Phase 7 — Observability & Metrics](#9-phase-7--observability--metrics)
10. [Phase 8 — Testing & Evaluation](#10-phase-8--testing--evaluation)
11. [Environment Variables Reference](#11-environment-variables-reference)
12. [Antigravity Prompts — Quick Reference](#12-antigravity-prompts--quick-reference)

---

## 1. Project Overview

### Problem
L2 support engineers waste significant time searching across ServiceNow tickets, KB articles, runbooks, and monitoring logs to resolve incidents. This increases MTTR and leads to inconsistent resolutions.

### Solution
A Retrieval-Augmented Generation (RAG) powered assistant that:
- Auto-analyses incoming ServiceNow incidents
- Retrieves semantically similar historical tickets, KB articles, and runbooks
- Generates context-aware root cause predictions, triage steps, and resolution drafts
- Surfaces recommendations inline within the ServiceNow engineer workspace
- Learns continuously from engineer feedback and resolved tickets

### Target Stack

| Layer | Technology |
|---|---|
| Backend runtime | Python 3.11, FastAPI |
| LLM | Azure OpenAI GPT-4o / Anthropic Claude Sonnet |
| Embedding model | `text-embedding-3-large` (OpenAI) or `BAAI/bge-large-en` |
| Vector database | Pinecone (primary) / pgvector (fallback) |
| RAG orchestration | LangChain v0.3 |
| ServiceNow integration | ServiceNow REST Table API, Webhooks |
| Frontend | ServiceNow UI Builder (Angular-based widget) |
| Task queue | Celery + Redis |
| Database | PostgreSQL 15 (feedback, audit logs, metadata) |
| Infrastructure | Docker, Kubernetes (AKS/EKS), GitHub Actions CI/CD |
| Monitoring | Prometheus + Grafana, LangSmith (LLM tracing) |

---

## 2. Repository & Project Structure

### Antigravity Prompt — Scaffold the project

```
Create a Python monorepo for a GenAI L2 Support Assistant with the following structure.
Use Python 3.11. Include placeholder files with module docstrings.

genai-l2-assistant/
├── README.md
├── .env.example
├── .gitignore                        # Python, Docker, secrets
├── docker-compose.yml                # local dev: API, Redis, Postgres, pgvector
├── pyproject.toml                    # Poetry project config
├── Makefile                          # shortcuts: make dev, make test, make lint
│
├── app/
│   ├── __init__.py
│   ├── main.py                       # FastAPI app entry point
│   ├── config.py                     # Pydantic settings from env vars
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── incidents.py          # POST /analyze, GET /similar
│   │   │   ├── recommendations.py   # GET /recommendations/{incident_id}
│   │   │   ├── feedback.py          # POST /feedback
│   │   │   ├── chat.py              # POST /chat (conversational interface)
│   │   │   └── health.py            # GET /health
│   │   └── middleware.py            # auth, RBAC, logging
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── rag_pipeline.py          # main RAG orchestration
│   │   ├── retriever.py             # hybrid BM25 + vector retrieval
│   │   ├── reranker.py              # cross-encoder reranking
│   │   ├── llm_client.py            # LLM abstraction (OpenAI / Anthropic)
│   │   ├── embedder.py              # embedding model wrapper
│   │   └── context_assembler.py     # token-aware context builder
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── servicenow_client.py     # ServiceNow REST API client
│   │   ├── ticket_processor.py      # NLP preprocessing pipeline
│   │   ├── kb_processor.py          # KB article chunking + embedding
│   │   ├── cmdb_enricher.py         # CMDB CI relationship enrichment
│   │   └── embedding_pipeline.py    # batch embedding + upsert to vector DB
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── incident.py              # Pydantic models: Incident, TicketChunk
│   │   ├── recommendation.py        # RecommendationResult, RootCause, Step
│   │   ├── feedback.py              # FeedbackRecord
│   │   └── chat.py                  # ChatMessage, ChatSession
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── vector_store.py          # Pinecone / pgvector abstraction
│   │   ├── postgres.py              # SQLAlchemy async session
│   │   └── cache.py                 # Redis cache client
│   │
│   ├── workers/
│   │   ├── __init__.py
│   │   ├── celery_app.py            # Celery configuration
│   │   ├── ingestion_worker.py      # async ticket ingestion task
│   │   └── reindex_worker.py        # nightly re-indexing task
│   │
│   ├── governance/
│   │   ├── __init__.py
│   │   ├── pii_anonymizer.py        # NER + regex PII masking
│   │   ├── rbac.py                  # role-based access enforcement
│   │   └── audit_logger.py          # structured audit event logger
│   │
│   └── utils/
│       ├── __init__.py
│       ├── text_utils.py            # chunking, cleaning, token counting
│       └── retry.py                 # exponential backoff decorator
│
├── servicenow/
│   ├── widget/                      # ServiceNow UI Builder widget files
│   │   ├── ai_sidebar.html
│   │   ├── ai_sidebar.js
│   │   └── ai_sidebar.css
│   ├── business_rules/
│   │   └── trigger_ai_analysis.js   # ServiceNow Business Rule script
│   └── flow_designer/
│       └── ai_analysis_flow.json    # Flow Designer export
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_rag_pipeline.py
│   │   ├── test_retriever.py
│   │   ├── test_pii_anonymizer.py
│   │   └── test_ticket_processor.py
│   ├── integration/
│   │   ├── test_servicenow_client.py
│   │   └── test_vector_store.py
│   └── eval/
│       ├── eval_retrieval.py        # retrieval precision/recall
│       └── eval_generation.py      # LLM output quality scoring
│
├── scripts/
│   ├── bootstrap_index.py           # one-time historical ticket indexing
│   ├── seed_test_data.py            # synthetic data for local dev
│   └── eval_run.py                  # run evaluation suite
│
├── infra/
│   ├── k8s/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── configmap.yaml
│   └── github-actions/
│       ├── ci.yml
│       └── deploy.yml
│
└── docs/
    ├── architecture.md
    ├── api-reference.md
    └── runbook.md
```

---

## 3. Phase 1 — Foundation & Environment Setup

### 3.1 Dependencies & Config

**Antigravity Prompt:**
```
In genai-l2-assistant/, set up the Python project using Poetry.

1. Create pyproject.toml with these dependencies:
   - fastapi==0.111.*
   - uvicorn[standard]==0.29.*
   - pydantic==2.*
   - pydantic-settings==2.*
   - langchain==0.3.*
   - langchain-openai==0.1.*
   - langchain-anthropic==0.1.*
   - langchain-pinecone==0.1.*
   - langchain-community==0.2.*
   - openai==1.*
   - anthropic==0.28.*
   - pinecone-client==3.*
   - pgvector==0.2.*
   - sqlalchemy[asyncio]==2.*
   - asyncpg==0.29.*
   - alembic==1.13.*
   - celery[redis]==5.*
   - redis==5.*
   - spacy==3.7.*
   - rank_bm25==0.2.*
   - sentence-transformers==2.*
   - httpx==0.27.*
   - tenacity==8.*
   - structlog==24.*
   - prometheus-fastapi-instrumentator==6.*
   - pytest==8.*
   - pytest-asyncio==0.23.*
   - pytest-mock==3.*

2. Create app/config.py using pydantic-settings with these settings classes:
   - LLMSettings: provider (openai|anthropic), model_name, api_key, azure_endpoint, azure_api_version
   - EmbeddingSettings: model_name, dimensions (default 3072)
   - VectorStoreSettings: provider (pinecone|pgvector), pinecone_api_key, pinecone_index_name, pinecone_environment
   - ServiceNowSettings: instance_url, username, password, client_id, client_secret
   - DatabaseSettings: postgres_url, redis_url
   - AppSettings: composes all above, reads from env with prefix handling

3. Create .env.example with all required keys and placeholder values.
   Add comments explaining each variable.

4. Create docker-compose.yml with services:
   - api: builds from ./Dockerfile, mounts app/, hot reload
   - postgres: postgres:15-alpine, with pgvector extension enabled
   - redis: redis:7-alpine
   - worker: same image as api, runs celery worker
```

### 3.2 Database Schema

**Antigravity Prompt:**
```
In app/storage/postgres.py and using Alembic, create the database schema.

Create SQLAlchemy async models for these tables:

1. incidents
   - id: UUID primary key
   - snow_sys_id: VARCHAR(32) unique (ServiceNow sys_id)
   - number: VARCHAR(20) (e.g. INC0042871)
   - short_description: TEXT
   - description: TEXT
   - category: VARCHAR(100)
   - subcategory: VARCHAR(100)
   - priority: SMALLINT (1-5)
   - state: VARCHAR(50)
   - assignment_group: VARCHAR(200)
   - assigned_to: VARCHAR(200)
   - cmdb_ci: VARCHAR(200) (impacted CI from CMDB)
   - opened_at: TIMESTAMP WITH TIME ZONE
   - resolved_at: TIMESTAMP WITH TIME ZONE nullable
   - resolution_notes: TEXT nullable
   - root_cause: TEXT nullable
   - is_indexed: BOOLEAN default false
   - created_at: TIMESTAMP WITH TIME ZONE server_default now()
   - updated_at: TIMESTAMP WITH TIME ZONE onupdate now()

2. recommendations
   - id: UUID primary key
   - incident_id: UUID FK → incidents.id
   - root_cause_prediction: TEXT
   - confidence_score: FLOAT (0.0–1.0)
   - triage_steps: JSONB (list of step objects)
   - similar_incidents: JSONB (list of {number, sys_id, similarity_score, resolution_summary})
   - kb_references: JSONB (list of {kb_number, title, relevance_score})
   - resolution_draft: TEXT
   - retrieval_latency_ms: INTEGER
   - generation_latency_ms: INTEGER
   - created_at: TIMESTAMP WITH TIME ZONE server_default now()

3. feedback
   - id: UUID primary key
   - recommendation_id: UUID FK → recommendations.id
   - incident_id: UUID FK → incidents.id
   - engineer_id: VARCHAR(200) (ServiceNow user sys_id)
   - rating: SMALLINT (1=thumbs_down, 5=thumbs_up)
   - comment: TEXT nullable
   - acted_on_steps: JSONB nullable (which steps the engineer followed)
   - created_at: TIMESTAMP WITH TIME ZONE server_default now()

4. audit_events
   - id: UUID primary key
   - event_type: VARCHAR(100) (ticket_analyzed, recommendation_served, feedback_submitted, chat_query)
   - actor_id: VARCHAR(200) (engineer user ID)
   - actor_role: VARCHAR(50) (l2_engineer, l3_engineer, admin)
   - resource_type: VARCHAR(50)
   - resource_id: VARCHAR(200)
   - payload: JSONB
   - ip_address: INET
   - created_at: TIMESTAMP WITH TIME ZONE server_default now()

5. chat_sessions
   - id: UUID primary key
   - incident_id: UUID FK → incidents.id
   - engineer_id: VARCHAR(200)
   - messages: JSONB (list of {role, content, timestamp})
   - created_at: TIMESTAMP WITH TIME ZONE
   - updated_at: TIMESTAMP WITH TIME ZONE

Generate the Alembic migration for all tables.
Add appropriate indexes: incidents(snow_sys_id), incidents(is_indexed), feedback(recommendation_id), audit_events(actor_id, created_at).
```

---

## 4. Phase 2 — Data Pipeline & Indexing

### 4.1 ServiceNow Client

**Antigravity Prompt:**
```
Create app/ingestion/servicenow_client.py — a fully async ServiceNow REST API client.

Requirements:
- Use httpx.AsyncClient with connection pooling
- Support OAuth 2.0 (client_credentials) and basic auth, configurable via settings
- Implement token refresh with automatic retry on 401
- Use tenacity for exponential backoff on transient failures (429, 503)
- All methods should be typed with Pydantic models

Implement these methods:

1. get_incident(sys_id: str) -> IncidentRecord
   GET /api/now/table/incident/{sys_id}
   Fields: number, short_description, description, category, subcategory, priority,
   state, assignment_group, assigned_to, cmdb_ci, opened_at, resolved_at,
   work_notes, resolution_notes, u_root_cause (custom field)

2. list_incidents(params: IncidentQueryParams) -> list[IncidentRecord]
   GET /api/now/table/incident with sysparm_query, sysparm_limit, sysparm_offset
   Support filters: state, assignment_group, opened_at_start, opened_at_end, category

3. list_kb_articles(params: KBQueryParams) -> list[KBArticle]
   GET /api/now/table/kb_knowledge
   Fields: number, short_description, text, category, valid_to, workflow_state

4. get_cmdb_ci(ci_sys_id: str) -> CMDBRecord
   GET /api/now/table/cmdb_ci/{sys_id}
   Fields: name, sys_class_name, operational_status, environment, u_service_tier,
   relationship_list (upstream/downstream services)

5. update_incident_work_note(sys_id: str, note: str) -> bool
   PATCH /api/now/table/incident/{sys_id}
   Body: {"work_notes": note}

6. update_incident_resolution(sys_id: str, resolution_notes: str, close_code: str) -> bool
   PATCH /api/now/table/incident/{sys_id}
   Body: {"resolution_notes": ..., "close_code": ..., "state": "6"}

Include a mock_client.py alongside it that returns fixture data from tests/fixtures/
for use in local development without a live ServiceNow instance.
```

### 4.2 NLP Preprocessing Pipeline

**Antigravity Prompt:**
```
Create app/ingestion/ticket_processor.py — the NLP preprocessing pipeline for incident tickets.

Use spaCy (en_core_web_trf model) and implement:

1. class TicketPreprocessor:

   def preprocess(self, incident: IncidentRecord) -> ProcessedTicket:
     Returns a ProcessedTicket with:
     - cleaned_text: combined short_description + description, lowercased,
       whitespace normalized, HTML stripped
     - entities: list of extracted entities with label (SERVICE, ERROR_CODE,
       HOSTNAME, IP_ADDRESS, APPLICATION, ENVIRONMENT)
     - keywords: top-15 TF-IDF keywords
     - category_vector: one-hot encoding of category + subcategory
     - summary: 2-sentence extractive summary using first + most informative sentence

   def mask_pii(self, text: str) -> str:
     Uses the PII anonymizer (import from governance/pii_anonymizer.py) before any
     text leaves the preprocessor. Always called before embedding.

   def classify_incident_type(self, text: str) -> str:
     Rule-based + keyword classifier returning one of:
     ["application_error", "infrastructure", "network", "security",
      "performance", "access_management", "data_issue", "unknown"]

2. class TicketChunker:
   def chunk(self, processed: ProcessedTicket, chunk_size: int = 512,
             overlap: int = 50) -> list[TextChunk]:
     Sentence-aware splitting. Each chunk includes:
     - chunk_text: the text content
     - chunk_type: "description" | "work_notes" | "resolution" | "kb_article"
     - source_id: incident number or KB article number
     - metadata: dict with category, priority, cmdb_ci, opened_at, resolved_at,
       incident_type, assignment_group

3. class KBArticleProcessor:
   def process(self, article: KBArticle) -> list[TextChunk]:
     Splits KB article body by section headers first, then by sentence-aware
     chunking. Preserves section title in each chunk's metadata.
```

### 4.3 PII Anonymizer

**Antigravity Prompt:**
```
Create app/governance/pii_anonymizer.py — PII detection and masking pipeline.

Implement class PIIAnonymizer with:

1. Pattern-based detection (compiled regex) for:
   - Email addresses → [EMAIL]
   - IP addresses (IPv4 + IPv6) → [IP_ADDRESS]
   - Hostnames matching common internal patterns (*.internal, *.corp, *.local) → [HOSTNAME]
   - Phone numbers (international formats) → [PHONE]
   - Credit card numbers (Luhn check) → [CARD_NUMBER]
   - AWS/GCP/Azure access keys (pattern-based) → [CLOUD_KEY]
   - UUIDs → retain (system identifiers, not PII)

2. NER-based detection using spaCy:
   - PERSON entities → [PERSON]
   - ORG entities in context of "reported by", "assigned to", "escalated by" → [PERSON]
   - GPE (location) entities only when in address context → [LOCATION]

3. def anonymize(self, text: str) -> AnonymizedResult:
   Returns both the masked text and a list of replacements made
   (type, original_length, position) for audit logging — never store originals.

4. def is_safe_to_index(self, text: str) -> tuple[bool, list[str]]:
   Returns (safe, reasons) — if high-confidence PII remains after anonymization,
   flag for human review rather than silently indexing.

Write unit tests in tests/unit/test_pii_anonymizer.py covering:
- Email in ticket description
- Hostname masking
- Person name in "assigned to John Smith" context
- Cloud key pattern
- Text with no PII (should pass through unchanged)
```

### 4.4 Embedding Pipeline

**Antigravity Prompt:**
```
Create app/ingestion/embedding_pipeline.py — batch embedding and vector store upsert.

Implement class EmbeddingPipeline:

1. __init__:
   - Initialise embedder from app/core/embedder.py
   - Initialise vector store from app/storage/vector_store.py
   - Set up structlog logger

2. async def run_batch(self, chunks: list[TextChunk], batch_size: int = 100) -> BatchResult:
   - Embed in batches of `batch_size` to respect rate limits
   - For each batch: embed → build vector records → upsert to vector store
   - Track progress, errors, total tokens used
   - Return BatchResult(total_chunks, indexed, failed, tokens_used, duration_s)

3. async def run_single(self, chunk: TextChunk) -> bool:
   - For real-time indexing of a newly resolved ticket
   - Embed single chunk and upsert immediately

4. Vector record format for Pinecone upsert:
   {
     "id": "{source_id}_{chunk_index}",
     "values": [float, ...],  // embedding vector
     "metadata": {
       "source_id": "INC0042871",
       "source_type": "incident" | "kb_article" | "runbook",
       "chunk_type": "description" | "resolution" | "work_notes",
       "category": "...",
       "subcategory": "...",
       "priority": 1-5,
       "cmdb_ci": "...",
       "assignment_group": "...",
       "incident_type": "...",
       "opened_at": "ISO8601",
       "resolved_at": "ISO8601 or null",
       "resolution_time_min": int or null,
       "text_preview": first 200 chars  // for display in UI
     }
   }

5. Create app/scripts/bootstrap_index.py:
   - Reads all resolved incidents from ServiceNow (last 2 years, state=Resolved/Closed)
   - Reads all active KB articles
   - Runs through TicketPreprocessor → TicketChunker → EmbeddingPipeline
   - Logs progress to stdout with ETA
   - Idempotent: skip already-indexed incidents (check incidents.is_indexed flag)
   - Supports --dry-run flag (process but don't upsert)
```

### 4.5 Vector Store Abstraction

**Antigravity Prompt:**
```
Create app/storage/vector_store.py — a provider-agnostic vector store interface.

1. Define abstract base class VectorStore with:
   - async upsert(records: list[VectorRecord]) -> UpsertResult
   - async query(vector: list[float], top_k: int, filter: dict) -> list[QueryMatch]
   - async delete(ids: list[str]) -> bool
   - async describe_index() -> IndexStats

2. Implement PineconeVectorStore(VectorStore):
   - Use pinecone-client v3 async API
   - Index name and namespace from settings
   - Metadata filtering support: filter by source_type, category, cmdb_ci,
     assignment_group, chunk_type
   - Handle Pinecone upsert batching (max 100 vectors per request)

3. Implement PGVectorStore(VectorStore):
   - Use pgvector extension with SQLAlchemy async
   - Cosine similarity search using <=> operator
   - Create embeddings table: id, source_id, source_type, embedding vector(3072),
     metadata jsonb, created_at
   - Support same metadata filtering via WHERE clause generation

4. Factory function get_vector_store(settings: VectorStoreSettings) -> VectorStore
   Returns correct implementation based on settings.provider

5. class QueryMatch(BaseModel):
   - id: str
   - score: float
   - metadata: dict
   - text_preview: str
```

---

## 5. Phase 3 — Core AI Engine (RAG)

### 5.1 Embedder

**Antigravity Prompt:**
```
Create app/core/embedder.py — embedding model wrapper with caching.

Implement class Embedder:

1. Support two providers via settings:
   a. OpenAI: use AsyncOpenAI client, model text-embedding-3-large, dimensions=3072
   b. HuggingFace: use sentence-transformers BAAI/bge-large-en-v1.5 locally

2. async def embed_text(self, text: str) -> list[float]:
   - Truncate to model's max token limit before embedding
   - Cache results in Redis with TTL=24h, key=sha256(text)[:16]
   - Return normalised vector (L2 norm = 1.0)

3. async def embed_batch(self, texts: list[str]) -> list[list[float]]:
   - Process in parallel with asyncio.gather (max concurrency 10)
   - Respect rate limits: implement token bucket at 1M tokens/min for OpenAI
   - Return list in same order as input

4. def count_tokens(self, text: str) -> int:
   - Use tiktoken for OpenAI, tokenizer for HF
   - Used by context assembler for budget management
```

### 5.2 Retriever

**Antigravity Prompt:**
```
Create app/core/retriever.py — hybrid BM25 + dense retrieval with metadata filtering.

Implement class HybridRetriever:

1. __init__:
   - Load BM25 index (built from chunk texts, serialised to Redis on first build)
   - Initialise vector store client
   - Initialise embedder

2. async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:

   RetrievalQuery fields:
   - query_text: str (incident description)
   - top_k: int = 20
   - filters: RetrievalFilters
     - source_types: list = ["incident", "kb_article", "runbook"]
     - categories: list[str] = []  // filter to same category if provided
     - cmdb_cis: list[str] = []    // filter to same/related CIs
     - min_resolution_date: datetime | None  // exclude very old resolutions
     - chunk_types: list = ["resolution", "description"]

   Steps:
   a. Embed query_text
   b. Dense retrieval: vector store query with top_k*3, apply metadata filters
   c. Sparse retrieval: BM25 top_k*2 from in-memory index
   d. Reciprocal Rank Fusion (RRF): merge dense + sparse results
      score = Σ 1/(k + rank_i) where k=60
   e. Return top_k after RRF, deduplicated by source_id+chunk_type

3. async def get_similar_incidents(
       self, incident: ProcessedTicket, top_n: int = 5
   ) -> list[SimilarIncident]:
   - Filter to source_type="incident", chunk_type="resolution"
   - Exclude the incident itself
   - Return SimilarIncident(number, sys_id, similarity_score, resolution_summary,
     resolution_time_min, category)

4. async def rebuild_bm25_index(self) -> bool:
   - Fetch all chunk texts from Postgres
   - Rebuild BM25 index and serialize to Redis
   - Called by nightly reindex worker
```

### 5.3 Context Assembler

**Antigravity Prompt:**
```
Create app/core/context_assembler.py — token-aware context window builder.

Implement class ContextAssembler:

MAX_CONTEXT_TOKENS = 6000  // budget for retrieved context (leaves room for prompt + response)

1. def assemble(
       self,
       incident: ProcessedTicket,
       retrieved_chunks: list[RetrievedChunk],
       max_tokens: int = MAX_CONTEXT_TOKENS
   ) -> AssembledContext:

   Strategy:
   a. Prioritise resolution chunks over description chunks (resolution = ground truth)
   b. Prioritise high similarity score (>0.85) chunks
   c. Deduplicate: if two chunks from same incident, keep higher-scored one
   d. Fill token budget greedily: add chunks in priority order until budget exhausted
   e. Always include at least 1 KB article chunk if available

   Return AssembledContext with:
   - incident_summary: str (2-sentence summary of the incoming incident)
   - similar_incidents_context: formatted block of top-5 similar resolved incidents
   - kb_context: formatted block of relevant KB article sections
   - total_tokens: int
   - sources_used: list[SourceReference]

2. def format_incident_context(self, chunks: list[RetrievedChunk]) -> str:
   Format as:
   ---
   SIMILAR INCIDENT: {number} (Resolved in {time}min, Similarity: {score:.0%})
   Issue: {description_preview}
   Resolution: {resolution_text}
   Root Cause: {root_cause if available}
   ---

3. def format_kb_context(self, chunks: list[RetrievedChunk]) -> str:
   Format as:
   ---
   KB ARTICLE: {number} — {title}
   {article_text}
   ---
```

### 5.4 LLM Client

**Antigravity Prompt:**
```
Create app/core/llm_client.py — LLM abstraction supporting OpenAI and Anthropic.

Implement class LLMClient:

1. Support providers: "openai" (Azure or direct) and "anthropic"
   Select at runtime from settings.llm.provider

2. async def generate(self, prompt: LLMPrompt) -> LLMResponse:
   LLMPrompt: system_prompt, user_message, temperature=0.1, max_tokens=2000
   LLMResponse: content, model, input_tokens, output_tokens, latency_ms

3. async def generate_streaming(self, prompt: LLMPrompt) -> AsyncIterator[str]:
   Yield tokens as they arrive (for future streaming UI support)

4. Implement retry with tenacity: max_attempts=3, wait_exponential(min=1, max=10)
   on RateLimitError and APIConnectionError

5. Log every call to LangSmith if LANGSMITH_API_KEY is set:
   - Input prompt, output, model, latency, tokens
   - Tag with incident_id if present in prompt metadata
```

### 5.5 RAG Pipeline — Core Orchestrator

**Antigravity Prompt:**
```
Create app/core/rag_pipeline.py — the main RAG orchestration pipeline.

Implement class RAGPipeline:

1. async def analyze_incident(
       self, incident_id: str, engineer_role: str
   ) -> RecommendationResult:

   Full pipeline:
   a. Load incident from DB (or fetch from ServiceNow if not yet stored)
   b. Preprocess with TicketPreprocessor (PII mask → classify → extract entities)
   c. Retrieve with HybridRetriever (top_k=20, filter by category + cmdb_ci)
   d. Assemble context with ContextAssembler
   e. Generate recommendations with LLM using RECOMMENDATION_PROMPT_TEMPLATE
   f. Parse LLM output into structured RecommendationResult
   g. Store recommendation in DB
   h. Log to audit trail
   i. Return RecommendationResult

2. RECOMMENDATION_PROMPT_TEMPLATE (system prompt):
"""
You are an expert L2 support assistant. Your role is to help engineers resolve
IT incidents quickly and accurately.

You will be given:
1. The current incident details
2. Similar resolved incidents from history
3. Relevant knowledge base articles

Based ONLY on the provided context, you must respond with a JSON object (no markdown,
no preamble) with this exact structure:
{
  "root_cause_prediction": "2-3 sentence explanation of the most likely root cause",
  "confidence_score": 0.0-1.0,
  "triage_steps": [
    {"step": 1, "action": "...", "rationale": "...", "command": "optional CLI/script"},
    ...
  ],
  "resolution_draft": "Draft resolution note ready to paste into ServiceNow work notes",
  "escalate_to_l3": true/false,
  "escalation_reason": "Only if escalate_to_l3 is true",
  "sources_used": ["INC0039201", "KB0012345"]
}

Rules:
- Only recommend actions supported by the provided context
- If confidence_score < 0.6, set escalate_to_l3 = true
- Triage steps must be specific and actionable, not generic
- Resolution draft must be in past tense as if engineer already resolved it
- Never invent commands or procedures not present in context
"""

3. async def chat(
       self, incident_id: str, message: str, session_id: str, engineer_role: str
   ) -> ChatResponse:

   Conversational follow-up on an existing recommendation:
   a. Load existing recommendation for incident
   b. Load chat history from DB for session_id
   c. Build CHAT_PROMPT with incident context + history + new message
   d. Generate response
   e. Append to session history
   f. Return ChatResponse(message, sources)

4. async def generate_resolution_draft(
       self, incident_id: str, resolution_summary: str
   ) -> str:
   Generates a formatted ServiceNow resolution note from a brief engineer summary,
   enriched with the recommendation context.
```

---

## 6. Phase 4 — ServiceNow Integration & API Layer

### 6.1 FastAPI Routes

**Antigravity Prompt:**
```
Create the FastAPI routes in app/api/routes/.

1. app/api/routes/incidents.py:

POST /api/v1/incidents/analyze
  Body: { "snow_sys_id": str, "engineer_id": str }
  - Fetch incident from ServiceNow if not in DB
  - Trigger RAG pipeline via Celery task (async) or inline if urgent
  - Return: { "recommendation_id": UUID, "status": "processing" | "complete", "result": RecommendationResult | null }

GET /api/v1/incidents/{snow_sys_id}/recommendation
  - Return cached recommendation if exists and < 30 min old
  - Otherwise trigger fresh analysis
  - Include cache-control headers

GET /api/v1/incidents/{snow_sys_id}/similar
  Query params: top_n=5, min_similarity=0.7
  - Return list of similar incidents with resolution summaries
  - Used for the "Similar incidents" panel

POST /api/v1/incidents/{snow_sys_id}/webhook
  - ServiceNow webhook endpoint called on ticket creation/assignment
  - Validate HMAC signature from ServiceNow (X-ServiceNow-Signature header)
  - Enqueue Celery task for async analysis
  - Return 202 Accepted immediately

2. app/api/routes/feedback.py:

POST /api/v1/feedback
  Body: { "recommendation_id": UUID, "rating": 1|5, "comment": str|null,
          "acted_on_steps": list[int]|null, "engineer_id": str }
  - Store feedback record
  - If rating == 1 (negative) and comment exists, flag for review queue
  - Return { "id": UUID, "status": "recorded" }

3. app/api/routes/chat.py:

POST /api/v1/chat
  Body: { "incident_id": str, "message": str, "session_id": str, "engineer_id": str }
  - Route to RAGPipeline.chat()
  - Return { "response": str, "sources": list, "session_id": str }

4. app/api/middleware.py:
  - RBACMiddleware: validate X-Engineer-Id and X-Engineer-Role headers
    (in production these come from ServiceNow session; in dev accept from header)
  - Request logging middleware: log method, path, status, latency via structlog
  - CORS middleware: allow ServiceNow instance domain

5. app/main.py:
  - Register all routers with /api/v1 prefix
  - Add Prometheus instrumentation (prometheus-fastapi-instrumentator)
  - Add /health endpoint returning { status, version, vector_store_ping, db_ping }
  - Startup event: verify DB connection, vector store ping, LLM ping
```

### 6.2 Celery Workers

**Antigravity Prompt:**
```
Create app/workers/ingestion_worker.py and app/workers/reindex_worker.py.

1. ingestion_worker.py:

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def analyze_incident_async(self, snow_sys_id: str, engineer_id: str):
  """
  Triggered by ServiceNow webhook on new/assigned ticket.
  Runs the full RAG pipeline and stores result in DB.
  """
  Steps:
  a. Fetch incident from ServiceNow
  b. Preprocess + mask PII
  c. Run RAG pipeline
  d. Store RecommendationResult in DB
  e. Push notification back to ServiceNow work notes:
     Add a work note: "AI Analysis complete. View recommendations in the AI sidebar."
  f. On failure: self.retry(exc=exc)

2. reindex_worker.py:

@celery_app.task
def nightly_reindex():
  """
  Runs nightly at 02:00 UTC (configured via Celery beat).
  Indexes all tickets resolved since last run.
  """
  Steps:
  a. Query DB for incidents resolved since last_indexed_at (stored in a config table)
  b. Run EmbeddingPipeline.run_batch() on new chunks
  c. Rebuild BM25 index in Redis
  d. Update last_indexed_at
  e. Log stats: new_tickets_indexed, total_index_size, duration_s

Configure Celery beat schedule in celery_app.py:
  - nightly_reindex: crontab(hour=2, minute=0)
  - bm25_rebuild: crontab(hour=3, minute=0)  // rebuild BM25 after reindex
```

### 6.3 ServiceNow Widget

**Antigravity Prompt:**
```
Create the ServiceNow UI Builder widget files in servicenow/widget/.

1. ai_sidebar.html — Angular template for the widget:
   Structure:
   - Header bar: "AI Analysis" label + status badge (Analysing... | Ready | Low confidence)
   - Root cause section:
     - Bold prediction text
     - Confidence bar (CSS progress bar, green > 0.8, amber 0.6-0.8, red < 0.6)
     - "Low confidence — consider L3 escalation" warning if score < 0.6
   - Triage steps section:
     - Ordered list of steps
     - Each step has: step number, action text, rationale (collapsible), optional command (monospace)
     - Checkbox to mark step complete
   - Similar incidents section:
     - List of up to 5 cards showing: incident number (link), resolution time, similarity %
   - KB references section:
     - List of KB articles with title and relevance score
   - Resolution draft section:
     - Pre-filled textarea with draft resolution note
     - "Copy to work notes" button
   - Ask AI section:
     - Text input with send button
     - Suggested prompts as pill buttons:
       "Likely cause?", "Show similar incidents", "Check runbook", "Escalation criteria"
   - Feedback section:
     - Thumbs up / thumbs down buttons
     - Optional comment textarea (shown after rating)

2. ai_sidebar.js — Widget controller:
   - On widget load: call backend POST /api/v1/incidents/analyze with current ticket sys_id
   - Poll GET /api/v1/incidents/{sys_id}/recommendation every 3s until status=complete
   - Implement copyToWorkNotes(): copies draft to ServiceNow work_notes field using
     g_form.setValue('work_notes', draft)
   - Implement sendChatMessage(): POST /api/v1/chat, append response to chat thread
   - Implement submitFeedback(rating): POST /api/v1/feedback
   - Get engineer context from NOW.user.userID and a custom role check function

3. ai_sidebar.css — Scoped styles matching ServiceNow baseline theme.

4. servicenow/business_rules/trigger_ai_analysis.js:
   Business Rule script (runs on incident insert + when assigned_to changes):
   - Table: incident
   - When: after insert, after update (condition: assigned_to changed)
   - Script: make HTTP POST to webhook endpoint with incident sys_id and HMAC signature
   - Include error handling and logging to syslog

Include comments in each file explaining how to import/deploy in ServiceNow.
```

---

## 7. Phase 5 — Frontend UI (ServiceNow Widget)

### 7.1 Local Dev Simulator

**Antigravity Prompt:**
```
Create a standalone HTML/JS simulator for the AI sidebar widget
at servicenow/widget/simulator.html.

This allows developers to test the widget UI without a live ServiceNow instance.

Requirements:
- Self-contained single HTML file (no build step)
- Simulates a ServiceNow incident form on the left, AI sidebar on the right
- Pre-populated with a sample P1 incident about a payment service 502 error
- Connects to the local FastAPI backend at http://localhost:8000
- Uses the same CSS as ai_sidebar.css
- Includes a "Mock Mode" toggle that returns fixture data without hitting the API
- Responsive: works at 1280px and 1440px viewport widths

Mock fixture data should include:
- root_cause_prediction with confidence 0.87
- 4 triage steps (one with a sample kubectl command)
- 3 similar incidents
- 2 KB article references
- A pre-written resolution draft
```

---

## 8. Phase 6 — Feedback Loop & Continuous Learning

**Antigravity Prompt:**
```
Create app/core/feedback_processor.py — feedback analysis and index weight adjustment.

Implement class FeedbackProcessor:

1. async def process_feedback_batch(self, since: datetime) -> FeedbackStats:
   Called nightly by Celery worker.
   
   a. Load all feedback records since `since`
   b. For each negative feedback (rating=1):
      - Identify which retrieved chunks were used in the recommendation
      - Record a "negative signal" against those chunk IDs in Redis
        (key: "neg_signals:{chunk_id}", increment counter)
   c. For each positive feedback (rating=5):
      - Record a "positive signal" against used chunk IDs
   d. Compute per-source quality score:
      score = (positive_signals) / (positive_signals + negative_signals + 1)
   e. Store scores in a feedback_weights table in Postgres:
      (source_id, source_type, quality_score, last_updated)
   f. Return FeedbackStats(total_feedback, positive_rate, sources_updated)

2. Modify HybridRetriever.retrieve() to apply feedback weights:
   Add a step after RRF: multiply each chunk's score by its quality_score from
   feedback_weights (default 1.0 if no feedback yet).
   This causes positively-reinforced sources to rank higher over time.

3. async def flag_for_review(self, recommendation_id: UUID, reason: str):
   Insert record into a review_queue table for human review.
   Triggered when: rating=1 AND confidence_score was > 0.8 (model was confident but wrong)
   These are the most valuable training signals.

4. Create a Celery task process_feedback_nightly() in workers/reindex_worker.py
   that calls process_feedback_batch() and logs stats.
```

---

## 9. Phase 7 — Observability & Metrics

**Antigravity Prompt:**
```
Set up observability across the application.

1. Prometheus metrics in app/core/rag_pipeline.py:
   Add these custom metrics using prometheus_client:
   
   - rag_analysis_duration_seconds (Histogram, labels: status)
   - rag_retrieval_duration_seconds (Histogram)
   - rag_generation_duration_seconds (Histogram)
   - recommendation_confidence_score (Histogram, buckets: 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
   - feedback_rating_total (Counter, labels: rating)
   - incidents_analyzed_total (Counter, labels: category, priority)
   - vector_store_query_duration_seconds (Histogram)
   - escalation_recommended_total (Counter)

2. Structured logging in all modules using structlog:
   Configure in app/main.py:
   - JSON output in production (LOG_FORMAT=json)
   - Human-readable in development
   - Auto-bind request_id, incident_id, engineer_id to all log lines in request context

3. Create infra/grafana/dashboards/l2_assistant.json:
   Grafana dashboard with panels:
   - Incidents analyzed per hour (timeseries)
   - P50/P95/P99 recommendation latency (timeseries)
   - Confidence score distribution (histogram)
   - Feedback positive rate over time (stat + timeseries)
   - Escalation rate (stat)
   - Vector store query latency (timeseries)
   - Error rate by endpoint (timeseries)

4. LangSmith tracing:
   In app/core/llm_client.py, wrap every LLM call with LangSmith RunTree:
   - Tag each run with: incident_id, category, confidence_score, recommendation_id
   - This enables prompt debugging and output quality review in LangSmith UI
```

---

## 10. Phase 8 — Testing & Evaluation

### 10.1 Unit Tests

**Antigravity Prompt:**
```
Create comprehensive unit tests in tests/unit/.

1. tests/unit/test_rag_pipeline.py:
   Test class TestRAGPipeline with pytest-asyncio:
   
   - test_analyze_incident_success: mock retriever + LLM, verify RecommendationResult structure
   - test_analyze_incident_low_confidence: confidence < 0.6 → escalate_to_l3=True
   - test_analyze_incident_llm_failure: LLM throws, verify graceful fallback response
   - test_chat_maintains_history: second chat message includes first exchange in context
   - test_pii_not_in_llm_prompt: verify PII masking applied before LLM sees text

2. tests/unit/test_retriever.py:
   - test_hybrid_retrieval_merges_results: mock dense + sparse, verify RRF scores
   - test_metadata_filter_applied: category filter passed to vector store query
   - test_deduplication: same source_id in dense + sparse → appears once in output

3. tests/unit/test_ticket_processor.py:
   - test_preprocess_extracts_error_codes: "ORA-00942" → entities includes ERROR_CODE
   - test_classify_incident_type_application_error
   - test_chunking_respects_sentence_boundaries
   - test_kb_chunking_preserves_section_titles

4. tests/conftest.py:
   Provide fixtures:
   - sample_incident_p1: ProcessedTicket fixture (payment service 502)
   - sample_recommendation: RecommendationResult fixture
   - mock_vector_store: AsyncMock of VectorStore
   - mock_llm_client: AsyncMock returning fixture JSON response
   - mock_servicenow_client: returns fixture incident data
```

### 10.2 Retrieval Evaluation

**Antigravity Prompt:**
```
Create tests/eval/eval_retrieval.py — offline retrieval quality evaluation.

Evaluation dataset format (tests/eval/data/retrieval_eval.jsonl):
Each line: { "query": "...", "expected_source_ids": ["INC0039201", "KB0012345"], "category": "..." }

Implement:

1. class RetrievalEvaluator:
   
   def evaluate(self, eval_dataset: list[EvalCase], top_k: int = 10) -> EvalReport:
   
   Metrics per query:
   - Precision@k: fraction of top-k results that are relevant
   - Recall@k: fraction of relevant results found in top-k
   - MRR (Mean Reciprocal Rank): 1/rank of first relevant result
   - NDCG@k: normalized discounted cumulative gain
   
   Aggregate metrics:
   - mean Precision@5, Precision@10
   - mean Recall@5, Recall@10
   - mean MRR
   - mean NDCG@10
   - breakdown by category

2. Create tests/eval/data/retrieval_eval.jsonl with 30 synthetic eval cases
   covering: application_error, infrastructure, network, access_management categories.
   Each case has a realistic incident query and 2-3 expected source IDs.

3. Create scripts/eval_run.py:
   CLI script: python scripts/eval_run.py --eval-type retrieval --output results.json
   Runs evaluation and prints a formatted report table.
   Fails (exit 1) if mean MRR < 0.5 (quality gate for CI).
```

### 10.3 Generation Evaluation

**Antigravity Prompt:**
```
Create tests/eval/eval_generation.py — LLM output quality evaluation.

Evaluation approach: LLM-as-judge using a separate GPT-4o call to score outputs.

1. class GenerationEvaluator:

   async def evaluate(self, test_cases: list[GenerationTestCase]) -> GenerationReport:
   
   GenerationTestCase: {
     incident_description: str,
     context: AssembledContext,
     generated_recommendation: RecommendationResult,
     reference_resolution: str  // actual resolution from historical ticket
   }
   
   Score each output on (1-5 scale, via LLM judge):
   - Groundedness: are claims supported by provided context? (no hallucination)
   - Relevance: are steps relevant to the specific incident?
   - Actionability: are steps specific and executable?
   - Accuracy: does root cause match the reference resolution?
   
   Judge prompt template:
   "Score the following AI-generated incident recommendation on {criterion}
   from 1-5. Respond with only the integer score and a one-sentence justification.
   
   Context provided to AI: {context}
   Generated output: {output}
   Reference resolution: {reference}
   
   Score for {criterion}:"

2. Quality gate thresholds (enforce in CI):
   - Groundedness >= 4.0 (hallucination prevention — critical)
   - Relevance >= 3.5
   - Actionability >= 3.5
   - Accuracy >= 3.0
```

---

## 11. Environment Variables Reference

```bash
# LLM Provider
LLM_PROVIDER=openai                            # openai | anthropic
LLM_MODEL_NAME=gpt-4o                          # model identifier
OPENAI_API_KEY=sk-...                          # OpenAI direct API key
AZURE_OPENAI_API_KEY=...                       # Azure OpenAI API key
AZURE_OPENAI_ENDPOINT=https://...              # Azure OpenAI endpoint
AZURE_OPENAI_API_VERSION=2024-02-01
ANTHROPIC_API_KEY=sk-ant-...

# Embedding
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=3072

# Vector Store
VECTOR_STORE_PROVIDER=pinecone                 # pinecone | pgvector
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=l2-assistant-index
PINECONE_ENVIRONMENT=us-east-1-aws

# ServiceNow
SNOW_INSTANCE_URL=https://your-instance.service-now.com
SNOW_USERNAME=svc_ai_assistant
SNOW_PASSWORD=...
SNOW_CLIENT_ID=...
SNOW_CLIENT_SECRET=...
SNOW_WEBHOOK_SECRET=...                        # HMAC secret for webhook validation

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/l2assistant
REDIS_URL=redis://localhost:6379/0

# App
APP_ENV=development                            # development | staging | production
LOG_FORMAT=human                               # human | json
SECRET_KEY=...                                 # FastAPI session secret
ALLOWED_ORIGINS=https://your-instance.service-now.com

# Observability
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=l2-assistant-prod
PROMETHEUS_ENABLED=true
```

---

## 12. Antigravity Prompts — Quick Reference

Below are the exact prompts to give Antigravity, in execution order:

```
PROMPT 1 — Project scaffold
"Scaffold the genai-l2-assistant project using the directory structure
and specifications in Phase 1 of the implementation plan."

PROMPT 2 — Dependencies and config
"Set up pyproject.toml with all dependencies and create app/config.py
with Pydantic settings classes as specified in section 3.1."

PROMPT 3 — Docker environment
"Create docker-compose.yml for local development with postgres (pgvector),
redis, api, and worker services as specified in section 3.1."

PROMPT 4 — Database schema and migrations
"Create SQLAlchemy async models and Alembic migrations for all tables
in section 3.2. Run alembic init and generate the initial migration."

PROMPT 5 — ServiceNow client
"Implement app/ingestion/servicenow_client.py as specified in section 4.1.
Include the MockServiceNowClient alongside it."

PROMPT 6 — NLP preprocessing
"Implement app/ingestion/ticket_processor.py (TicketPreprocessor,
TicketChunker, KBArticleProcessor) as specified in section 4.2."

PROMPT 7 — PII anonymizer + tests
"Implement app/governance/pii_anonymizer.py and its unit tests
as specified in section 4.3."

PROMPT 8 — Vector store abstraction
"Implement app/storage/vector_store.py with PineconeVectorStore
and PGVectorStore as specified in section 4.5."

PROMPT 9 — Embedding pipeline
"Implement app/ingestion/embedding_pipeline.py and
scripts/bootstrap_index.py as specified in section 4.4."

PROMPT 10 — Core AI components
"Implement app/core/embedder.py, app/core/retriever.py,
app/core/context_assembler.py, and app/core/llm_client.py
as specified in sections 5.1–5.4."

PROMPT 11 — RAG pipeline orchestrator
"Implement app/core/rag_pipeline.py including the full analyze_incident()
and chat() methods with all prompt templates as specified in section 5.5."

PROMPT 12 — FastAPI routes and middleware
"Implement all FastAPI routes in app/api/routes/ and middleware
as specified in section 6.1."

PROMPT 13 — Celery workers
"Implement app/workers/ingestion_worker.py and app/workers/reindex_worker.py
with Celery beat schedule as specified in section 6.2."

PROMPT 14 — ServiceNow widget
"Create the ServiceNow UI widget files in servicenow/widget/ and the
Business Rule script as specified in section 6.3."

PROMPT 15 — Widget simulator
"Create the standalone local dev simulator at servicenow/widget/simulator.html
as specified in section 7.1."

PROMPT 16 — Feedback processor
"Implement app/core/feedback_processor.py with quality score tracking
and feedback-weighted retrieval as specified in section 8."

PROMPT 17 — Observability
"Add Prometheus metrics, structured logging, and Grafana dashboard JSON
as specified in section 9."

PROMPT 18 — Unit tests
"Create all unit tests in tests/unit/ with fixtures in tests/conftest.py
as specified in section 10.1."

PROMPT 19 — Retrieval evaluation
"Create tests/eval/eval_retrieval.py with the synthetic eval dataset
and eval_run.py CLI script as specified in section 10.2."

PROMPT 20 — Generation evaluation
"Create tests/eval/eval_generation.py with the LLM-as-judge scoring
and quality gates as specified in section 10.3."
```

---

## Appendix A — Data Flow Summary

```
ServiceNow ticket created
        │
        ▼
Webhook → POST /api/v1/incidents/{sys_id}/webhook
        │
        ▼
Celery task: analyze_incident_async(snow_sys_id)
        │
        ├─► ServiceNow API → fetch full incident record
        │
        ├─► TicketPreprocessor → clean, PII mask, classify, extract entities
        │
        ├─► HybridRetriever → BM25 + vector search → RRF merge → top-20 chunks
        │
        ├─► ContextAssembler → token-aware context (similar incidents + KB articles)
        │
        ├─► LLMClient → generate(RECOMMENDATION_PROMPT + context + incident)
        │
        ├─► Parse JSON → RecommendationResult
        │
        ├─► Store in DB (recommendations table)
        │
        └─► ServiceNow work note: "AI Analysis ready"

Engineer opens ticket in ServiceNow
        │
        ▼
AI sidebar widget loads → GET /api/v1/incidents/{sys_id}/recommendation
        │
        ▼
Display: root cause (confidence bar) + triage steps + similar incidents + KB refs
        │
        ├─► Engineer checks off steps, copies resolution draft
        │
        ├─► Engineer asks follow-up → POST /api/v1/chat
        │
        └─► Engineer submits feedback → POST /api/v1/feedback
                │
                └─► FeedbackProcessor updates quality scores nightly
```

## Appendix B — Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| RAG vs fine-tuning | RAG | No retraining cost; retrieves from live incident data; auditable sources |
| Hybrid retrieval | BM25 + dense | BM25 catches exact error codes; dense catches semantic similarity |
| Sync vs async analysis | Async (Celery) | Webhook returns 202 immediately; engineers not blocked |
| Chunking strategy | Sentence-aware, 512 tokens, 50 overlap | Preserves resolution step integrity; overlap avoids cut-off context |
| Confidence threshold | 0.6 | Below this, L3 escalation recommended; tunable per service category |
| PII masking timing | Before embedding | PII never enters vector index or LLM prompt |
| Feedback signal | Per-chunk quality score | Enables granular source-level learning without full retraining |
| Widget placement | ServiceNow UI Builder sidebar | Inline with ticket; no context switching for engineer |

---

*Last updated: 2026-06-24 | Version: 1.0.0*
*For questions on this implementation plan, refer to the architecture overview in docs/architecture.md*
