from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

metrics_router = APIRouter(tags=["Metrics"])


@metrics_router.get("/metrics")
async def prometheus_metrics() -> Response:
    """Every metric in Prometheus' text format, for a scraper (or a load test) to read."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
