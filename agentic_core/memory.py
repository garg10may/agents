"""
Shared and per-agent memory/workspace
"""
class Memory:
    def __init__(self):
        self._store = {}
    def get(self, key, default=None):
        return self._store.get(key, default)
    def set(self, key, value):
        self._store[key] = value
    def append(self, key, value):
        self._store.setdefault(key, []).append(value)
    def all(self):
        return dict(self._store)
