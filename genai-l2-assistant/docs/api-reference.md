# API Reference

Base URL: `https://<host>:8000`

All endpoints return JSON. Authentication is via API key in the `X-API-Key` header or ServiceNow webhook HMAC signature.

---

## Health & Status

### GET /health

Health check endpoint for load balancers and Kubernetes probes.

**Response 200:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "production",
  "timestamp": "2024-01-15T14:30:00Z"
}
```

### GET /metrics

Prometheus metrics endpoint (when `PROMETHEUS_ENABLED=true`).

**Response 200:** Prometheus text format with application metrics.

---

## Incident Analysis

### POST /api/v1/analyze

Analyze an incident and generate an AI recommendation.

**Request Body:**
```json
{
  "sys_id": "abc123def456abc123def456abc12345",
  "number": "INC0042871",
  "short_description": "Payment service returning 502 errors",
  "description": "The payment-service has been returning intermittent HTTP 502...",
  "category": "application",
  "subcategory": "availability",
  "priority": 1,
  "cmdb_ci": "payment-service",
  "assignment_group": "L2-Application-Support",
  "work_notes": "Confirmed 502s in Datadog APM..."
}
```

**Response 200:**
```json
{
  "id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
  "incident_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "snow_sys_id": "abc123def456abc123def456abc12345",
  "root_cause_prediction": "The HTTP 502 errors are caused by connection pool exhaustion...",
  "confidence_score": 0.87,
  "triage_steps": [
    {
      "step": 1,
      "action": "Check connection pool metrics in Datadog",
      "rationale": "Confirm pool exhaustion and identify leak pattern",
      "command": "kubectl exec -it deploy/payment-service -- curl localhost:8080/actuator/metrics"
    },
    {
      "step": 2,
      "action": "Review v2.4.1 changelog for connection handling changes",
      "rationale": "Identify the specific code change that caused the regression",
      "command": null
    }
  ],
  "resolution_draft": "Root cause: Connection pool exhaustion after v2.4.1 deployment...",
  "escalate_to_l3": false,
  "escalation_reason": null,
  "similar_incidents": [
    {
      "number": "INC0039201",
      "similarity_score": 0.92,
      "resolution_summary": "Connection pool tuning resolved 502 errors",
      "resolution_time_min": 45,
      "category": "application"
    }
  ],
  "kb_references": [
    {
      "kb_number": "KB0012345",
      "title": "Troubleshooting HTTP 502 Errors",
      "relevance_score": 0.91
    }
  ],
  "retrieval_latency_ms": 142,
  "generation_latency_ms": 1834,
  "created_at": "2024-01-15T14:35:00Z"
}
```

**Response 422:** Validation error (invalid request body).

**Response 500:** Internal server error.

---

## Webhook

### POST /api/v1/webhook/servicenow

ServiceNow webhook endpoint for real-time incident notifications.

**Headers:**
| Header | Description |
|--------|-------------|
| `X-ServiceNow-Signature` | HMAC-SHA256 signature of the request body |
| `Content-Type` | `application/json` |

**Request Body:**
```json
{
  "event_type": "incident.created",
  "sys_id": "abc123def456abc123def456abc12345",
  "number": "INC0042871",
  "short_description": "Payment service returning 502 errors",
  "description": "...",
  "category": "application",
  "priority": 1,
  "state": "1",
  "cmdb_ci": "payment-service"
}
```

**Response 202:**
```json
{
  "status": "accepted",
  "task_id": "d4e5f6a7-b8c9-0123-defa-234567890123"
}
```

**Response 401:** Invalid webhook signature.

---

## Chat

### POST /api/v1/chat

Send a follow-up chat message about an incident.

**Request Body:**
```json
{
  "incident_id": "abc123def456abc123def456abc12345",
  "message": "Should I rollback to v2.4.0 or increase the pool size?",
  "session_id": "d4e5f6a7-b8c9-0123-defa-234567890123",
  "engineer_id": "user-sys-id-001"
}
```

**Response 200:**
```json
{
  "response": "Based on similar incidents, I recommend increasing the pool size first as a quick mitigation, then investigating the connection leak. INC0039201 was resolved in 45 minutes with this approach. If the error rate doesn't decrease within 15 minutes, proceed with a rollback to v2.4.0.",
  "sources": ["INC0039201", "KB0012345"],
  "session_id": "d4e5f6a7-b8c9-0123-defa-234567890123"
}
```

---

## Feedback

### POST /api/v1/feedback

Submit engineer feedback on an AI recommendation.

**Request Body:**
```json
{
  "recommendation_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
  "rating": 5,
  "comment": "Root cause was spot on, triage steps saved 30 minutes",
  "acted_on_steps": [1, 2, 3],
  "engineer_id": "user-sys-id-001"
}
```

**Response 200:**
```json
{
  "id": "c3d4e5f6-a7b8-9012-cdef-123456789012",
  "status": "recorded"
}
```

### GET /api/v1/feedback/stats

Get aggregated feedback statistics.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `days` | int | Look-back period in days (default: 30) |

**Response 200:**
```json
{
  "total_feedback": 847,
  "positive_count": 712,
  "negative_count": 135,
  "positive_rate": 0.84,
  "sources_updated": 523
}
```

---

## Incidents

### GET /api/v1/incidents

List incidents from the local database.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | string | Filter by state (e.g., "1", "2", "6") |
| `category` | string | Filter by category |
| `assignment_group` | string | Filter by assignment group |
| `limit` | int | Max results (default: 100, max: 1000) |
| `offset` | int | Pagination offset (default: 0) |

**Response 200:**
```json
{
  "incidents": [...],
  "total": 1247,
  "limit": 100,
  "offset": 0
}
```

### GET /api/v1/incidents/{sys_id}

Get a single incident by ServiceNow sys_id.

**Response 200:** Full incident record.

**Response 404:** Incident not found.

---

## Error Responses

All error responses follow this format:

```json
{
  "detail": {
    "error": "error_code",
    "message": "Human-readable error description",
    "timestamp": "2024-01-15T14:30:00Z"
  }
}
```

| Status Code | Description |
|-------------|-------------|
| 400 | Bad request (malformed input) |
| 401 | Unauthorized (invalid API key or webhook signature) |
| 404 | Resource not found |
| 422 | Validation error |
| 429 | Rate limit exceeded |
| 500 | Internal server error |
| 503 | Service unavailable (LLM or vector store down) |
