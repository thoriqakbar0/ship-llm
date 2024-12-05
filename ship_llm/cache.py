from collections import OrderedDict
from typing import Any, Optional
import hashlib
import json

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, moving it to the end if found"""
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any) -> None:
        """Add item to cache, removing oldest item if at capacity"""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

def create_cache_key(data: Any) -> str:
    """Create a stable cache key from any serializable data"""
    try:
        key_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    except TypeError as e:
        print(f"Warning: Could not create cache key: {str(e)}")
        return None
