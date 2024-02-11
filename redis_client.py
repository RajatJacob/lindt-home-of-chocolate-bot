import redis
from datetime import timedelta
from typing import Callable
from functools import wraps

redis_client: redis.Redis | None = None


def get_redis_client():
    global redis_client
    if redis_client is None:
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
    return redis_client


def redis_cache(fn: Callable[..., str]) -> Callable[..., str]:
    @wraps(fn)
    def wrapper(*args, **kwargs) -> str:
        client = get_redis_client()
        key = f"{fn.__name__}:{args}:{kwargs}"
        if client.exists(key):
            cached = client.get(key)
            if type(cached) is bytes:
                return str(cached.decode('utf-8'))
            return str(cached)
        result = fn(*args, **kwargs)
        client.set(key, result, ex=timedelta(days=30))
        return str(result)
    return wrapper
