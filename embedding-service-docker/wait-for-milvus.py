import os
import sys
import time
import urllib.error
import urllib.request


health_url = os.getenv("MILVUS_HEALTH_URL", "http://milvus-standalone:9091/healthz")
timeout_seconds = float(os.getenv("MILVUS_WAIT_TIMEOUT_SECONDS", "300"))
interval_seconds = float(os.getenv("MILVUS_WAIT_INTERVAL_SECONDS", "5"))
deadline = time.monotonic() + timeout_seconds

while True:
    try:
        with urllib.request.urlopen(health_url, timeout=5) as response:
            if 200 <= response.status < 300:
                break
    except (OSError, urllib.error.URLError) as exc:
        print(f"Waiting for Milvus at {health_url}: {exc}", flush=True)

    if time.monotonic() >= deadline:
        print(f"Milvus did not become healthy within {timeout_seconds:g} seconds", file=sys.stderr)
        sys.exit(1)
    time.sleep(interval_seconds)

print("Milvus is healthy; starting embedding service", flush=True)
os.execvp("python", ["python", "-u", "/app/embedding_service.py"])
