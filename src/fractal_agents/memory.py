import json
import os
from typing import Any, Dict, Optional

import lz4.frame
import redis


class FractalMemory:
    def __init__(self, redis_url: Optional[str] = None, use_compression: bool = True):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.client = redis.Redis.from_url(self.redis_url, decode_responses=False)  # Binary safe
        self.prefix = "fractal:node:"
        self.use_compression = use_compression

    def _get_key(self, node_id: str) -> str:
        return f"{self.prefix}{node_id}"

    def save_node_state(self, node_id: str, state: Dict[str, Any]):
        """Saves the entire state of a node."""
        json_str = json.dumps(state)
        data = json_str.encode("utf-8")

        if self.use_compression:
            data = lz4.frame.compress(data)

        self.client.set(self._get_key(node_id), data)

    def get_node_state(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the state of a node."""
        data = self.client.get(self._get_key(node_id))
        if not data:
            return None

        try:
            if self.use_compression:
                # Try to decompress, if it fails it might be uncompressed legacy data
                try:
                    data = lz4.frame.decompress(data)
                except Exception:
                    # Fallback for uncompressed data
                    pass

            return json.loads(data.decode("utf-8"))
        except Exception:
            return None

    def store_summary(self, node_id: str, summary: str):
        """Stores just the summary for quick retrieval by siblings/children."""
        # Summaries are usually small, no compression needed
        self.client.hset("fractal:summaries", node_id, summary)

    def get_summary(self, node_id: str) -> str:
        val = self.client.hget("fractal:summaries", node_id)
        if val:
            return val.decode("utf-8") if isinstance(val, bytes) else val
        return ""
