"""Prometheus metrics, scraped from /metrics.

Every model metric carries the model version, so two versions serving at once can be told apart.
Labels stay bounded: a version, a status, one of 77 intent names; never anything from a request.
"""

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("intent_api_requests_total", "Prediction requests answered", ["version", "status"])
IN_FLIGHT = Gauge("intent_api_requests_in_flight", "Prediction requests being processed (saturation)")
INFERENCE_SECONDS = Histogram(
    "intent_api_inference_seconds", "Time inside the model per request (tokenizing included)", ["version"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
PREDICTED_INTENTS = Counter(
    "intent_api_predicted_intent_total", "Top intent per request: the output distribution", ["version", "intent"]
)
TOP_CONFIDENCE = Histogram(
    "intent_api_top_confidence", "Probability of the top intent; a shift here is an early drift signal",
    ["version"], buckets=[i / 10 for i in range(1, 11)],
)
