# GenAI L2 Support Assistant

RAG-powered AI assistant that helps L2 support engineers resolve ServiceNow incidents faster by providing intelligent root cause analysis, triage recommendations, and resolution drafts — all based on historical incident data and knowledge base articles.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ServiceNow Instance                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────────────────────┐ │
│  │ Incident │  │ KB       │  │ AI Sidebar Widget                    │ │
│  │ Table    │  │ Articles │  │ (Root cause, triage, similar tickets)│ │
│  └────┬─────┘  └────┬─────┘  └──────────────────▲───────────────────┘ │
│       │              │                           │                     │
└───────┼──────────────┼───────────────────────────┼─────────────────────┘
        │ Webhook      │ REST API                  │ REST API
        ▼              ▼                           │
┌───────────────────────────────────────────────────┼─────────────────────┐
│                  GenAI L2 Assistant Backend       │                     │
│                                                   │                     │
│  ┌──────────┐    ┌────────────────┐    ┌─────────┴────────┐           │
│  │ FastAPI  │───▶│ RAG Pipeline   │───▶│ Recommendation   │           │
│  │ /api/v1  │    │                │    │ Result           │           │
│  └──────────┘    │ 1. PII Mask   │    └──────────────────┘           │
│                  │ 2. Retrieve    │                                    │
│  ┌──────────┐    │ 3. Rerank     │    ┌──────────────────┐           │
│  │ Celery   │───▶│ 4. Assemble   │    │ Feedback         │           │
│  │ Workers  │    │ 5. LLM Gen    │    │ Processor        │           │
│  └──────────┘    │ 6. Parse      │    └──────────────────┘           │
│                  └───────┬────────┘                                    │
│                          │                                             │
│         ┌────────────────┼────────────────┐                           │
│         ▼                ▼                ▼                            │
│  ┌────────────┐  ┌────────────┐  ┌────────────────┐                  │
│  │ Pinecone / │  │ PostgreSQL │  │ Redis          │                  │
│  │ pgvector   │  │ (metadata, │  │ (cache, BM25,  │                  │
│  │ (vectors)  │  │  feedback) │  │  Celery broker)│                  │
│  └────────────┘  └────────────┘  └────────────────┘                  │
│                                                                       │
│  ┌────────────────┐  ┌────────────────┐                              │
│  │ OpenAI GPT-4o  │  │ Anthropic      │   (LLM Providers)           │
│  │ / Azure OpenAI │  │ Claude Sonnet  │                              │
│  └────────────────┘  └────────────────┘                              │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Description |
|---------|-------------|
| **AI Root Cause Analysis** | Generates root cause predictions with confidence scores (0.0–1.0) |
| **Intelligent Triage Steps** | Actionable resolution steps with optional CLI commands |
| **Similar Incident Matching** | Hybrid BM25 + vector retrieval with Reciprocal Rank Fusion |
| **KB Article References** | Links relevant knowledge base articles with relevance scores |
| **Conversational Follow-up** | Multi-turn chat for incident deep-dives |
| **PII Protection** | Masks emails, IPs, names, cloud keys *before* embedding & LLM |
| **Feedback Loop** | Engineer ratings drive per-source quality scoring |
| **Confidence Escalation** | Low-confidence (<0.6) predictions auto-recommend L3 escalation |

---

## First-Time Setup (Windows)

### Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11+ | [python.org](https://www.python.org/downloads/) |
| Docker Desktop | Latest | [docker.com](https://www.docker.com/products/docker-desktop/) |
| Git | Latest | [git-scm.com](https://git-scm.com/) |

### Step 1: Clone & Configure Environment

```powershell
# Clone the repository
git clone https://github.com/your-org/genai-l2-assistant.git
cd genai-l2-assistant

# Copy the environment template
cp .env.example .env
```

Now edit `.env` with your actual API keys:

```env
# REQUIRED — Pick one LLM provider:
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-4o
OPENAI_API_KEY=sk-your-key-here

# REQUIRED — Embedding model (uses same OpenAI key):
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=3072

# REQUIRED — Vector store (pick one):
VECTOR_STORE_PROVIDER=pinecone          # or "pgvector" for local
PINECONE_API_KEY=your-pinecone-key      # only if using pinecone
PINECONE_INDEX_NAME=l2-assistant-index
PINECONE_ENVIRONMENT=us-east-1-aws

# OPTIONAL — ServiceNow (skip for local dev, mock client is used):
# SNOW_INSTANCE_URL=https://your-instance.service-now.com
# SNOW_USERNAME=svc_ai_assistant
# SNOW_PASSWORD=your-password

# Database (defaults work with Docker Compose):
DATABASE_URL=postgresql+asyncpg://l2assistant:l2assistant@localhost:5432/l2assistant
REDIS_URL=redis://localhost:6379/0
```

> **Tip:** For a fully local setup without cloud APIs, set `VECTOR_STORE_PROVIDER=pgvector` — pgvector runs inside the Docker Postgres container.

### Step 2: Create Virtual Environment & Install Dependencies

```powershell
# Create virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Install the project in editable mode (so "app.*" imports work)
pip install -e .

# Install dev dependencies (for running tests)
pip install -r requirements-dev.txt
```

### Step 3: Start Infrastructure (PostgreSQL + Redis)

```powershell
# Start Docker Desktop first, then:
docker-compose up -d postgres redis
```

Verify services are running:
```powershell
docker-compose ps
# Should show postgres and redis as "Up (healthy)"
```

### Step 4: Run Database Migrations

```powershell
alembic upgrade head
```

This creates all database tables: `incidents`, `recommendations`, `feedback`, `audit_events`, `chat_sessions`, `feedback_weights`, `review_queue`.

### Step 5: Seed Test Data

```powershell
python scripts/seed_test_data.py
```

This generates:
- **10 synthetic incidents** across 5 categories (application, infrastructure, network, access management, security)
- **5 KB articles** covering common troubleshooting scenarios
- All with realistic descriptions, work notes, resolution notes, and root causes

### Step 6: Start the API Server

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Verify it's running:
- **Swagger UI**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health

---

## Data Pipeline — Indexing & Vectorization

The assistant needs a vector index of historical incidents and KB articles to perform retrieval. Here's how the data pipeline works:

### How Vectorization Works

```
 ServiceNow Incidents / KB Articles
              │
              ▼
 ┌──────────────────────────┐
 │ 1. FETCH                 │  ServiceNow REST API (or mock client)
 │    Get incidents & KB    │
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ 2. PREPROCESS            │  TicketPreprocessor
 │    • Strip HTML          │  • Clean & normalize text
 │    • Extract entities    │  • Classify incident type
 │    • Extract keywords    │  • Generate summary
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ 3. PII MASK              │  PIIAnonymizer
 │    • Emails → [EMAIL]    │  • IPs → [IP_ADDRESS]
 │    • Names → [PERSON]    │  • Cloud keys → [CLOUD_KEY]
 │    PII never reaches     │  the vector store or LLM.
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ 4. CHUNK                 │  TicketChunker / KBArticleProcessor
 │    Sentence-aware split  │  512 tokens per chunk, 50 token overlap
 │    Types: description,   │  work_notes, resolution, kb_article
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ 5. EMBED                 │  Embedder (OpenAI text-embedding-3-large)
 │    Generate 3072-dim     │  vectors for each chunk
 │    Cached in Redis       │  (SHA-256 key, 24h TTL)
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ 6. UPSERT                │  VectorStore (Pinecone / pgvector)
 │    Store vectors with    │  metadata (category, priority,
 │    cmdb_ci, chunk_type)  │  for filtered retrieval
 └──────────────────────────┘
```

### Bootstrap — Index Historical Data (One-Time)

Run this once to index your existing resolved incidents and KB articles:

```powershell
# Full run (fetches from ServiceNow, processes, embeds, upserts)
python scripts/bootstrap_index.py

# Dry run (processes but doesn't upsert — good for testing)
python scripts/bootstrap_index.py --dry-run

# With custom batch size
python scripts/bootstrap_index.py --batch-size 50 --limit 500
```

> **Without ServiceNow access?** The mock client will be used automatically in development mode, providing 5 sample incidents for indexing.

### Real-Time Indexing

When an incident is resolved in ServiceNow:

1. **Webhook fires** → `POST /api/v1/incidents/{sys_id}/webhook`
2. **Celery task** picks it up asynchronously
3. **EmbeddingPipeline.run_single()** indexes the new ticket immediately
4. The incident is immediately available for future retrieval

### Nightly Reindex (Automatic)

Celery beat runs these tasks automatically:

| Time (UTC) | Task | Description |
|------------|------|-------------|
| 02:00 | `nightly_reindex` | Index all tickets resolved since last run |
| 03:00 | `rebuild_bm25` | Rebuild BM25 sparse index from Postgres |
| 04:00 | `process_feedback` | Update quality scores from engineer feedback |

To start the scheduler:
```powershell
celery -A app.workers.celery_app beat --loglevel=info
```

### Vector Store Setup

**Option A: Pinecone (recommended for production)**

1. Create a free account at [pinecone.io](https://www.pinecone.io/)
2. Create an index:
   - **Name:** `l2-assistant-index`
   - **Dimensions:** `3072`
   - **Metric:** `cosine`
3. Set in `.env`:
   ```env
   VECTOR_STORE_PROVIDER=pinecone
   PINECONE_API_KEY=your-key
   PINECONE_INDEX_NAME=l2-assistant-index
   ```

**Option B: pgvector (fully local, no cloud needed)**

Already included in the Docker Compose Postgres container. Just set:
```env
VECTOR_STORE_PROVIDER=pgvector
```

---

## Starting Workers (for Async Processing)

```powershell
# Start Celery worker (processes analysis tasks)
celery -A app.workers.celery_app worker --loglevel=info --pool=solo

# Start Celery beat (scheduled tasks — run in a separate terminal)
celery -A app.workers.celery_app beat --loglevel=info
```

> **Note:** On Windows, use `--pool=solo` for the Celery worker.

---

## Using the Widget Simulator

A standalone HTML simulator lets you test the AI sidebar without ServiceNow:

```powershell
# Open in your default browser
start servicenow/widget/simulator.html
```

The simulator includes:
- Pre-populated P1 payment service incident
- **Mock Mode** (on by default) with realistic fixture data
- Toggle Mock Mode OFF to connect to your local API at `http://localhost:8000`
- Full AI sidebar: root cause, triage steps, similar incidents, KB articles, chat, feedback

---

## Running Tests

```powershell
# Unit tests
pytest tests/unit/ -v --tb=short

# With coverage
pytest tests/unit/ -v --cov=app --cov-report=html

# Specific test file
pytest tests/unit/test_pii_anonymizer.py -v

# Retrieval evaluation (requires vector store to be populated)
python scripts/eval_run.py --eval-type retrieval --output results.json
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/incidents/analyze` | Trigger AI analysis for an incident |
| `GET` | `/api/v1/incidents/{sys_id}/recommendation` | Get cached recommendation |
| `GET` | `/api/v1/incidents/{sys_id}/similar` | Find similar incidents |
| `POST` | `/api/v1/incidents/{sys_id}/webhook` | ServiceNow webhook receiver |
| `POST` | `/api/v1/chat` | Conversational follow-up |
| `POST` | `/api/v1/feedback` | Submit engineer feedback |
| `GET` | `/health` | Health check (DB + Redis status) |
| `GET` | `/docs` | Swagger UI |

Full API reference: [docs/api-reference.md](docs/api-reference.md)

---

## Project Structure

```
genai-l2-assistant/
├── app/
│   ├── api/routes/          # FastAPI endpoints (incidents, chat, feedback, health)
│   ├── core/                # RAG engine (retriever, embedder, LLM client, pipeline)
│   ├── governance/          # PII anonymizer, RBAC, audit logging
│   ├── ingestion/           # ServiceNow client, ticket processor, embedding pipeline
│   ├── models/              # Pydantic data models
│   ├── storage/             # PostgreSQL ORM, vector store, Redis cache
│   ├── workers/             # Celery tasks (analysis, reindex, feedback)
│   ├── utils/               # Text chunking, retry helpers
│   └── config.py            # Pydantic settings from env vars
├── servicenow/
│   ├── widget/              # AI sidebar widget (HTML/JS/CSS) + simulator
│   └── business_rules/      # ServiceNow webhook trigger script
├── tests/
│   ├── unit/                # 38 unit tests
│   ├── integration/         # Integration test stubs
│   └── eval/                # Retrieval & generation evaluation suite
├── scripts/                 # CLI tools (bootstrap, seed, eval)
├── docs/                    # Architecture, API reference, runbook
├── infra/                   # Kubernetes, CI/CD, Grafana dashboards
├── alembic/                 # Database migrations
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Dev/test dependencies
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No module named 'app'` | Run `pip install -e .` to install the project |
| `make` not found on Windows | Use commands directly (e.g., `pytest` instead of `make test`) |
| Docker "daemon not running" | Start Docker Desktop and wait for it to fully load |
| `langchain-pinecone` install fails | Already removed — we use `pinecone-client` directly |
| Celery worker crashes on Windows | Use `--pool=solo` flag |
| `alembic upgrade head` fails | Ensure Postgres is running: `docker-compose up -d postgres` |

---

## Documentation

- [Architecture Overview](docs/architecture.md) — System diagram, data flow, design decisions
- [API Reference](docs/api-reference.md) — All endpoints with request/response examples
- [Operations Runbook](docs/runbook.md) — Deployment, monitoring, troubleshooting, scaling

---

## License

Proprietary — Internal use only.
