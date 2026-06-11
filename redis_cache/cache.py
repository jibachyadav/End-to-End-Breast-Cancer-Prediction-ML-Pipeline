import redis
import pandas as pd
import pyarrow as pa
import os

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB   = int(os.getenv("REDIS_DB", 0))
TTL        = 60 * 60 * 24  # 24 hours


def _get_client():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)


def to_redis(df, key):

    client = _get_client()
    table  = pa.Table.from_pandas(df)
    sink   = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, table.schema)
    writer.write_table(table)
    writer.close()
    client.set(key, bytes(sink.getvalue()), ex=TTL)
    print(f"[Redis] Saved '{key}' — {len(df)} rows")


def from_redis(key):
    
    client = _get_client()
    data   = client.get(key)
    if data is None:
        return None
    reader = pa.ipc.open_stream(pa.py_buffer(data))
    df     = reader.read_all().to_pandas()
    print(f"[Redis] Loaded '{key}' — {len(df)} rows")
    return df


def delete_key(key):
    _get_client().delete(key)


def flush_pipeline_keys():
    client = _get_client()
    keys   = client.keys("bc_*")
    if keys:
        client.delete(*keys)
        print(f"[Redis] Flushed {len(keys)} pipeline keys")