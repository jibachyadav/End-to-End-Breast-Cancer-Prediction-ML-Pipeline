cat > ~/End-to-End-Breast-Cancer-Prediction-ML-Pipeline/redis/cache.py << 'EOF'
"""
Redis Cache Utility
====================
Saves and loads Pandas DataFrames between Airflow pipeline stages
using Redis as a fast in-memory message bus.
"""

import redis
import pandas as pd
import pyarrow as pa
import os

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB   = int(os.getenv("REDIS_DB",   0))
TTL        = 60 * 60 * 24  # 24 hours

def _get_client() -> redis.Redis:
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

def to_redis(df: pd.DataFrame, key: str) -> None:
    client = _get_client()
    sink   = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, pa.Schema.from_pandas(df))
    writer.write_table(pa.Table.from_pandas(df))
    writer.close()
    client.set(key, bytes(sink.getvalue()), ex=TTL)
    print(f"[Redis] ✅ Saved '{key}' — {len(df)} rows, {df.shape[1]} cols")

def from_redis(key: str) -> pd.DataFrame:
    client = _get_client()
    data   = client.get(key)
    if data is None:
        raise KeyError(f"[Redis] ❌ Key '{key}' not found")
    reader = pa.ipc.open_stream(pa.py_buffer(data))
    df     = reader.read_all().to_pandas()
    print(f"[Redis] ✅ Loaded '{key}' — {len(df)} rows, {df.shape[1]} cols")
    return df

def delete_key(key: str) -> None:
    _get_client().delete(key)
    print(f"[Redis] 🗑️  Deleted '{key}'")

def flush_pipeline_keys() -> None:
    client = _get_client()
    keys   = client.keys("breast_cancer_*")
    if keys:
        client.delete(*keys)
        print(f"[Redis] 🗑️  Flushed {len(keys)} pipeline keys")
EOF