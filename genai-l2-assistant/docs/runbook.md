# Operations Runbook

## Table of Contents

- [Deployment](#deployment)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Scaling](#scaling)
- [Disaster Recovery](#disaster-recovery)

---

## Deployment

### Prerequisites

- Kubernetes 1.28+ cluster
- PostgreSQL 15+ with pgvector extension
- Redis 7+
- Pinecone account (or pgvector for self-hosted vector store)
- OpenAI API key or Azure OpenAI deployment
- ServiceNow instance with REST API access

### Initial Deployment

```bash
# 1. Create namespace
kubectl create namespace l2-assistant

# 2. Create secrets
kubectl create secret generic l2-assistant-secrets \
  --namespace l2-assistant \
  --from-literal=OPENAI_API_KEY=sk-... \
  --from-literal=PINECONE_API_KEY=... \
  --from-literal=SNOW_PASSWORD=... \
  --from-literal=SNOW_CLIENT_SECRET=... \
  --from-literal=SNOW_WEBHOOK_SECRET=... \
  --from-literal=DATABASE_URL=postgresql+asyncpg://... \
  --from-literal=SECRET_KEY=$(openssl rand -hex 32)

# 3. Apply Kubernetes manifests
kubectl apply -f infra/k8s/configmap.yaml
kubectl apply -f infra/k8s/deployment.yaml
kubectl apply -f infra/k8s/service.yaml

# 4. Verify deployment
kubectl get pods -n l2-assistant
kubectl logs -f deploy/l2-assistant-api -n l2-assistant

# 5. Run database migrations
kubectl exec -it deploy/l2-assistant-api -n l2-assistant -- alembic upgrade head

# 6. Bootstrap historical data
kubectl exec -it deploy/l2-assistant-api -n l2-assistant -- \
  python scripts/bootstrap_index.py --batch-size 100
```

### Rolling Updates

```bash
# Update deployment with new image
kubectl set image deploy/l2-assistant-api \
  l2-assistant-api=ghcr.io/org/l2-assistant:v1.2.0 \
  -n l2-assistant

# Monitor rollout
kubectl rollout status deploy/l2-assistant-api -n l2-assistant

# Rollback if needed
kubectl rollout undo deploy/l2-assistant-api -n l2-assistant
```

### Docker Compose (Development)

```bash
# Start all services
docker-compose up -d

# Run migrations
docker-compose exec api alembic upgrade head

# Seed test data
docker-compose exec api python scripts/seed_test_data.py

# View logs
docker-compose logs -f api
```

---

## Configuration

### Environment Variables

All configuration is via environment variables. See `.env.example` for the complete list.

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_PROVIDER` | No | `openai` or `anthropic` (default: `openai`) |
| `LLM_MODEL_NAME` | No | Model identifier (default: `gpt-4o`) |
| `OPENAI_API_KEY` | Yes* | OpenAI API key |
| `PINECONE_API_KEY` | Yes* | Pinecone API key |
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `REDIS_URL` | Yes | Redis connection string |
| `SNOW_INSTANCE_URL` | Yes | ServiceNow instance URL |
| `SNOW_WEBHOOK_SECRET` | Yes | HMAC secret for webhook validation |
| `APP_ENV` | No | `development`, `staging`, `production` |
| `SECRET_KEY` | Yes | Application secret key |

*Required based on provider selection.

### Tuning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_DIMENSIONS` | 3072 | Embedding vector dimensions |
| `EMBEDDING_MODEL` | text-embedding-3-large | Embedding model |
| Retriever `top_k` | 5 | Number of context chunks to retrieve |
| Retriever `rrf_k` | 60 | RRF constant |
| Chunker `chunk_size` | 512 | Max tokens per chunk |
| Chunker `overlap` | 50 | Overlap tokens between chunks |
| Confidence threshold | 0.6 | Below this → auto-escalate to L3 |

---

## Monitoring

### Health Checks

| Endpoint | Purpose | Expected |
|----------|---------|----------|
| `GET /health` | Liveness probe | `{"status": "healthy"}` |
| `GET /metrics` | Prometheus metrics | Prometheus text format |

### Key Metrics (Prometheus)

| Metric | Type | Description |
|--------|------|-------------|
| `l2_incidents_analyzed_total` | Counter | Total incidents analyzed |
| `l2_recommendation_latency_seconds` | Histogram | End-to-end recommendation latency |
| `l2_retrieval_latency_seconds` | Histogram | Retrieval step latency |
| `l2_generation_latency_seconds` | Histogram | LLM generation latency |
| `l2_confidence_score` | Histogram | Distribution of confidence scores |
| `l2_escalation_total` | Counter | L3 escalations triggered |
| `l2_feedback_total` | Counter | Feedback submissions (by rating) |
| `l2_feedback_positive_rate` | Gauge | Positive feedback rate |
| `l2_llm_errors_total` | Counter | LLM API errors |
| `l2_pii_masked_total` | Counter | PII patterns masked |

### Grafana Dashboard

Import the dashboard from `infra/grafana/dashboards/l2_assistant.json`.

**Key Panels:**
- Incidents per hour (rate)
- Latency P50/P95/P99
- Confidence score distribution
- Feedback rate (positive vs negative)
- Escalation rate
- Error rate by type

### Alerting Rules

| Alert | Condition | Severity |
|-------|-----------|----------|
| High Error Rate | Error rate > 5% for 5 min | Critical |
| High Latency | P99 latency > 10s for 5 min | Warning |
| Low Confidence | Avg confidence < 0.5 for 1 hour | Warning |
| High Escalation Rate | Escalation rate > 30% for 1 hour | Warning |
| LLM Unavailable | LLM errors > 50% for 2 min | Critical |
| Database Connection Pool | Active connections > 80% | Warning |

---

## Troubleshooting

### Common Issues

#### 1. LLM API Rate Limiting

**Symptoms:** 429 errors in logs, increased latency.

**Diagnosis:**
```bash
kubectl logs deploy/l2-assistant-api -n l2-assistant | grep "rate_limit"
```

**Resolution:**
- Check OpenAI usage dashboard for quota
- Increase retry backoff in configuration
- Consider switching to a higher-tier API plan
- Enable request queuing in Redis

#### 2. Vector Store Query Timeouts

**Symptoms:** Retrieval latency spikes, timeout errors.

**Diagnosis:**
```bash
# Check Pinecone index stats
curl -H "Api-Key: $PINECONE_API_KEY" \
  "https://$INDEX_HOST/describe_index_stats"
```

**Resolution:**
- Check Pinecone status page for outages
- Verify network connectivity to Pinecone endpoints
- Review index size and consider partitioning
- Fall back to pgvector if persistent

#### 3. Database Connection Pool Exhaustion

**Symptoms:** `asyncpg.exceptions.TooManyConnectionsError`.

**Diagnosis:**
```sql
SELECT count(*) FROM pg_stat_activity WHERE datname = 'l2assistant';
SELECT * FROM pg_stat_activity WHERE state = 'active';
```

**Resolution:**
- Increase `pool_size` in database settings (default: 20)
- Terminate idle connections
- Check for connection leaks in application code
- Increase PostgreSQL `max_connections`

#### 4. PII Leaking to LLM

**Symptoms:** Audit log shows PII in LLM prompts.

**Diagnosis:**
```sql
SELECT * FROM audit_events
WHERE event_type = 'llm_prompt_sent'
AND payload::text LIKE '%@%'
ORDER BY created_at DESC LIMIT 10;
```

**Resolution:**
- Review PII patterns in `app/governance/pii_anonymizer.py`
- Add missing patterns for new PII types
- Enable stricter PII mode in configuration
- Audit recent prompts and report any data exposure

#### 5. Webhook Signature Validation Failures

**Symptoms:** 401 responses to ServiceNow webhook calls.

**Diagnosis:**
```bash
kubectl logs deploy/l2-assistant-api -n l2-assistant | grep "webhook.*401"
```

**Resolution:**
- Verify `SNOW_WEBHOOK_SECRET` matches ServiceNow Business Rule configuration
- Check for whitespace or encoding issues in the secret
- Ensure ServiceNow is sending the `X-ServiceNow-Signature` header
- Test with a manual curl request

---

## Scaling

### Horizontal Scaling

```bash
# Scale API replicas
kubectl scale deploy/l2-assistant-api --replicas=4 -n l2-assistant

# Autoscaling based on CPU
kubectl autoscale deploy/l2-assistant-api \
  --min=2 --max=8 --cpu-percent=70 \
  -n l2-assistant
```

### Vertical Scaling

Update resource limits in `infra/k8s/deployment.yaml`:

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "1Gi"
    cpu: "1000m"
```

### Scaling Thresholds

| Metric | Scale Up | Scale Down |
|--------|----------|------------|
| CPU Utilization | > 70% | < 30% |
| Memory Usage | > 80% | < 40% |
| Request Queue Depth | > 100 | < 10 |
| P95 Latency | > 5s | < 1s |

### Database Scaling

- **Read replicas**: Add PostgreSQL read replicas for query scaling
- **Connection pooling**: Use PgBouncer for connection multiplexing
- **Partitioning**: Partition audit_events table by month

### Vector Store Scaling

- **Pinecone**: Scale pods via Pinecone dashboard
- **pgvector**: Add read replicas, tune `ivfflat` index parameters

---

## Disaster Recovery

### Backup Schedule

| Component | Frequency | Retention |
|-----------|-----------|-----------|
| PostgreSQL | Daily full + hourly WAL | 30 days |
| Pinecone | Managed by Pinecone | N/A |
| Redis | RDB snapshot every 15 min | 7 days |
| Configuration | Git versioned | Indefinite |

### Recovery Procedures

#### Database Recovery

```bash
# Restore from backup
pg_restore -d l2assistant /backups/l2assistant_20240115.dump

# Verify data integrity
psql -d l2assistant -c "SELECT count(*) FROM incidents;"
psql -d l2assistant -c "SELECT count(*) FROM recommendations;"
```

#### Vector Store Re-indexing

```bash
# Full re-index from database
python scripts/bootstrap_index.py --batch-size 100

# Verify index stats
curl -H "Api-Key: $PINECONE_API_KEY" \
  "https://$INDEX_HOST/describe_index_stats"
```

### RTO/RPO Targets

| Component | RTO | RPO |
|-----------|-----|-----|
| API Service | 5 min | 0 (stateless) |
| PostgreSQL | 30 min | 1 hour |
| Vector Store | 60 min | 24 hours |
| Redis Cache | 5 min | 15 min |
