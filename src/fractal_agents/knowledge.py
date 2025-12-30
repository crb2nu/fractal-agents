from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import uuid
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

class KnowledgeNode:
    def __init__(self, content: str, metadata: Dict[str, Any] = None, parent_id: Optional[str] = None, depth: int = 0):
        self.id = str(uuid.uuid4())
        self.content = content
        self.metadata = metadata or {}
        self.parent_id = parent_id
        self.depth = depth

class KnowledgeStore(ABC):
    @abstractmethod
    def add_node(self, node: KnowledgeNode, vector: List[float]):
        pass

    @abstractmethod
    def search(self, vector: List[float], parent_id: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        pass

class QdrantFractalStore(KnowledgeStore):
    def __init__(self, url: str = None, api_key: str = None, collection_name: str = "fractal_knowledge"):
        self.url = url or os.getenv("QDRANT_URL", "http://192.168.50.176:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.client = QdrantClient(url=self.url, api_key=self.api_key)
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(size=1536, distance=rest.Distance.COSINE), # Assuming OpenAI size
            )

    def add_node(self, node: KnowledgeNode, vector: List[float]):
        payload = {
            "content": node.content,
            "parent_id": node.parent_id,
            "depth": node.depth,
            **node.metadata
        }
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                rest.PointStruct(
                    id=node.id,
                    vector=vector,
                    payload=payload
                )
            ]
        )

    def search(self, vector: List[float], parent_id: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        filter_conditions = []
        if parent_id:
            filter_conditions.append(rest.FieldCondition(key="parent_id", match=rest.MatchValue(value=parent_id)))
        else:
            # If no parent_id, we usually query top-level (depth 0)
            filter_conditions.append(rest.FieldCondition(key="depth", match=rest.MatchValue(value=0)))

        query_filter = rest.Filter(must=filter_conditions) if filter_conditions else None

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=query_filter,
            limit=top_k
        )
        
        return [
            {
                "id": r.id,
                "content": r.payload["content"],
                "score": r.score,
                "payload": r.payload
            }
            for r in results
        ]

class FractalKnowledgeGraph:
    """
    Implements recursive retrieval logic across the fractal store.
    """
    def __init__(self, store: KnowledgeStore, llm_client: Any):
        self.store = store
        self.llm = llm_client

    def _get_embedding(self, text: str) -> List[float]:
        # Using the OpenAI-compatible client from our LLM interface
        # Note: In a real app, you'd use LiteLLM's embedding endpoint
        response = self.llm.client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def query(self, query_text: str, max_depth: int = 2, threshold: float = 0.7) -> str:
        vector = self._get_embedding(query_text)
        
        # Start recursive retrieval
        knowledge_chunks = []
        self._recursive_search(vector, None, 0, max_depth, threshold, knowledge_chunks)
        
        return "\n\n".join(knowledge_chunks)

    def _recursive_search(self, vector, parent_id, current_depth, max_depth, threshold, results_accumulator):
        if current_depth > max_depth:
            return

        hits = self.store.search(vector, parent_id=parent_id, top_k=3)
        for hit in hits:
            if hit["score"] >= threshold:
                results_accumulator.append(f"[Depth {current_depth}] {hit['content']}")
                # Zoom in: Search for children of this relevant node
                self._recursive_search(vector, hit["id"], current_depth + 1, max_depth, threshold, results_accumulator)
