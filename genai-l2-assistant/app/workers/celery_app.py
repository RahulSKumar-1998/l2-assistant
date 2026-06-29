"""Celery application configuration.

Creates the Celery app instance with Redis as both broker and result backend.
Configures task routing, serialization, and the Beat schedule for nightly jobs.
"""

import os

from celery import Celery
from celery.schedules import crontab

# Read Redis URL from environment with fallback for local development
_REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
_BROKER_URL: str = os.environ.get("CELERY_BROKER_URL", _REDIS_URL)
_BACKEND_URL: str = os.environ.get("CELERY_RESULT_BACKEND", _REDIS_URL)

def _is_redis_available(url_str: str) -> bool:
    import socket
    from urllib.parse import urlparse
    try:
        url = urlparse(url_str)
        host = url.hostname or "localhost"
        port = url.port or 6379
        s = socket.create_connection((host, port), timeout=0.5)
        s.close()
        return True
    except Exception:
        return False

_REDIS_AVAILABLE = _is_redis_available(_REDIS_URL)

# ── Celery App Instance ─────────────────────────────────────────────────────

celery_app = Celery(
    "l2_assistant",
    broker=_BROKER_URL if _REDIS_AVAILABLE else "memory://",
    backend=_BACKEND_URL if _REDIS_AVAILABLE else "cache+memory://",
)

# ── Configuration ───────────────────────────────────────────────────────────

celery_app.conf.update(
    task_always_eager=not _REDIS_AVAILABLE,
    task_eager_propagates=not _REDIS_AVAILABLE,
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,  # Results expire after 1 hour

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Task execution
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=200,
    worker_concurrency=4,

    # Task tracking
    task_track_started=True,
    task_send_sent_event=True,

    # Retry defaults
    task_default_retry_delay=60,  # 60 seconds
    task_default_max_retries=3,

    # Task routes — route tasks to dedicated queues
    task_routes={
        "app.workers.ingestion_worker.analyze_incident_async": {
            "queue": "analysis",
        },
        "app.workers.ingestion_worker.index_resolved_ticket": {
            "queue": "indexing",
        },
        "app.workers.reindex_worker.nightly_reindex": {
            "queue": "maintenance",
        },
        "app.workers.reindex_worker.rebuild_bm25": {
            "queue": "maintenance",
        },
        "app.workers.reindex_worker.process_feedback_nightly": {
            "queue": "maintenance",
        },
    },

    # Default queue for unrouted tasks
    task_default_queue="default",
)

# ── Beat Schedule ───────────────────────────────────────────────────────────

celery_app.conf.beat_schedule = {
    "nightly_reindex": {
        "task": "app.workers.reindex_worker.nightly_reindex",
        "schedule": crontab(hour=2, minute=0),  # 02:00 UTC
        "options": {"queue": "maintenance"},
    },
    "bm25_rebuild": {
        "task": "app.workers.reindex_worker.rebuild_bm25",
        "schedule": crontab(hour=3, minute=0),  # 03:00 UTC
        "options": {"queue": "maintenance"},
    },
    "process_feedback": {
        "task": "app.workers.reindex_worker.process_feedback_nightly",
        "schedule": crontab(hour=4, minute=0),  # 04:00 UTC
        "options": {"queue": "maintenance"},
    },
}

# ── Auto-discover Tasks ─────────────────────────────────────────────────────

celery_app.autodiscover_tasks(
    [
        "app.workers.ingestion_worker",
        "app.workers.reindex_worker",
    ]
)
