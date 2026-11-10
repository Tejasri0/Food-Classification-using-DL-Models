"""
seed_redis.py — one-off migration: load data/*.json into Redis
=================================================================

app.py reads nutrition facts and per-model metrics straight from Redis on
every "Predict" click; it does not read data/*.json at request time. Run
this script once (and again any time you edit the JSON files) to push their
contents into Redis under the keys app.py expects.

Usage
-----
    pip install redis
    python seed_redis.py

Respects the same REDIS_URL / REDIS_KEY_PREFIX env vars as app.py:
    REDIS_URL=redis://localhost:6379/0
    REDIS_KEY_PREFIX=food_classifier
"""

import json
import os
import sys

import redis

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Change this:
REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
REDIS_KEY_PREFIX = os.environ.get("REDIS_KEY_PREFIX", "food_classifier")

MODEL_KEYS = ("custom_cnn", "vgg16", "resnet50")

# (local filename, redis key, required)
FILES_TO_SEED = [
    ("nutritional_facts.json", f"{REDIS_KEY_PREFIX}:nutrition_facts", True),
    ("class_indices.json", f"{REDIS_KEY_PREFIX}:class_indices", False),
] + [
    (f"metrics_{key}.json", f"{REDIS_KEY_PREFIX}:metrics:{key}", True)
    for key in MODEL_KEYS
]


def main():
    client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        client.ping()
    except redis.RedisError as exc:
        print(f"Could not connect to Redis at {REDIS_URL}: {exc}", file=sys.stderr)
        sys.exit(1)

    seeded, missing = 0, []
    for filename, key, required in FILES_TO_SEED:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            (missing if required else missing).append(filename)
            if required:
                print(f"  ✗ {filename} not found (required) — skipping {key}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)  # validates it's real JSON before we push it

        client.set(key, json.dumps(data))
        print(f"  ✓ {filename} -> {key} ({len(data)} entries)")
        seeded += 1

    print(f"\nSeeded {seeded}/{len(FILES_TO_SEED)} keys into {REDIS_URL}.")
    if missing:
        print(f"Missing/skipped: {', '.join(missing)}")


if __name__ == "__main__":
    main()