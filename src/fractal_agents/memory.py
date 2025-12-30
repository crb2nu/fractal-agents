import json
import redis
import os
from typing import Dict, Any, Optional

class FractalMemory:
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        self.prefix = "fractal:node:"

    def _get_key(self, node_id: str) -> str:
        return f"{self.prefix}{node_id}"

    def save_node_state(self, node_id: str, state: Dict[str, Any]):
        """Saves the entire state of a node."""
        # Flattening for Hash storage or just dumping JSON for simplicity
        # Using JSON string for complex nested structures in a single key for now
        self.client.set(self._get_key(node_id), json.dumps(state))

    def get_node_state(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the state of a node."""
        data = self.client.get(self._get_key(node_id))
        if data:
            return json.loads(data)
        return None

    def store_summary(self, node_id: str, summary: str):
        """Stores just the summary for quick retrieval by siblings/children."""
        self.client.hset("fractal:summaries", node_id, summary)

    def get_summary(self, node_id: str) -> str:
        return self.client.hget("fractal:summaries", node_id) or ""
