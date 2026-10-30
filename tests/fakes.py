"""Test doubles shared by the worker and API tests."""


class FakeRedis:
    def __init__(self):
        self.kv, self.ttl, self.jobs = {}, {}, []
        self._ids = set()

    async def get(self, key):
        return self.kv.get(key)

    async def set(self, key, value, ex=None):
        self.kv[key], self.ttl[key] = value, ex

    async def enqueue_job(self, name, *args, _job_id=None, **kw):
        if _job_id in self._ids:
            return None
        self._ids.add(_job_id)
        self.jobs.append((name, args, _job_id))
        return object()

    async def ping(self):
        return True
