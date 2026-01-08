import os
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


class KnowledgeNode:
    def __init__(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        parent_id: Optional[str] = None,
        depth: int = 0,
    ):
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
    def search(
        self, vector: List[float], parent_id: Optional[str] = None, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        pass


class QdrantFractalStore(KnowledgeStore):
    def __init__(
        self, url: str = None, api_key: str = None, collection_name: str = "fractal_knowledge"
    ):
        self.url = url or os.getenv("QDRANT_URL", "http://192.168.50.176:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        print(f"[DEBUG] Qdrant URL: {self.url}")
        print(f"[DEBUG] Qdrant API Key present: {bool(self.api_key)}")
        self.client = QdrantClient(url=self.url, api_key=self.api_key, check_compatibility=False)
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=1536, distance=rest.Distance.COSINE
                ),  # Assuming OpenAI size
            )

    def add_node(self, node: KnowledgeNode, vector: List[float]):
        payload = {
            "content": node.content,
            "parent_id": node.parent_id,
            "depth": node.depth,
            **node.metadata,
        }
        self.client.upsert(
            collection_name=self.collection_name,
            points=[rest.PointStruct(id=node.id, vector=vector, payload=payload)],
        )

    def search(
        self, vector: List[float], parent_id: Optional[str] = None, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        filter_conditions = []
        if parent_id:
            filter_conditions.append(
                rest.FieldCondition(key="parent_id", match=rest.MatchValue(value=parent_id))
            )
        else:
            # If no parent_id, we usually query top-level (depth 0)
            filter_conditions.append(
                rest.FieldCondition(key="depth", match=rest.MatchValue(value=0))
            )

        query_filter = rest.Filter(must=filter_conditions) if filter_conditions else None

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=query_filter,
            limit=top_k,
        )

        return [
            {"id": r.id, "content": r.payload["content"], "score": r.score, "payload": r.payload}
            for r in results
        ]


class FractalKnowledgeGraph:
    """
    Implements recursive retrieval and indexing logic across the fractal store.

    Provides:
    - Recursive depth-first retrieval following parent-child relationships
    - Automatic embedding generation via LiteLLM
    - Node indexing with hierarchical structure preservation
    - Context summarization for retrieved knowledge
    """

    def __init__(self, store: KnowledgeStore, llm_client: Any):
        self.store = store
        self.llm = llm_client

    async def _get_embedding_async(self, text: str) -> List[float]:
        """Async embedding generation using the LLM client."""
        # Use the async client for embeddings
        response = await self.llm.client.embeddings.create(
            input=[text], model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def _get_embedding(self, text: str) -> List[float]:
        """Sync embedding generation (legacy compatibility)."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create a new task
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self._get_embedding_async(text))
                    return future.result()
            else:
                return loop.run_until_complete(self._get_embedding_async(text))
        except RuntimeError:
            return asyncio.run(self._get_embedding_async(text))

    async def index_node(
        self,
        content: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        depth: int = 0,
    ) -> str:
        """
        Indexes a knowledge node with automatic embedding generation.

        Returns the node ID for use as a parent in child nodes.
        """
        node = KnowledgeNode(
            content=content, metadata=metadata or {}, parent_id=parent_id, depth=depth
        )

        vector = await self._get_embedding_async(content)
        self.store.add_node(node, vector)

        return node.id

    async def index_hierarchy(
        self, nodes: List[Dict[str, Any]], parent_id: Optional[str] = None, base_depth: int = 0
    ) -> List[str]:
        """
        Indexes a hierarchy of nodes.

        Each node dict should have:
        - content: str
        - children: List[Dict] (optional)
        - metadata: Dict (optional)
        """
        indexed_ids = []

        for node_data in nodes:
            content = node_data.get("content", "")
            metadata = node_data.get("metadata", {})
            children = node_data.get("children", [])

            node_id = await self.index_node(
                content=content, parent_id=parent_id, metadata=metadata, depth=base_depth
            )
            indexed_ids.append(node_id)

            # Recursively index children
            if children:
                child_ids = await self.index_hierarchy(
                    children, parent_id=node_id, base_depth=base_depth + 1
                )
                indexed_ids.extend(child_ids)

        return indexed_ids

    def query(self, query_text: str, max_depth: int = 2, threshold: float = 0.7) -> str:
        """
        Performs recursive retrieval from the knowledge graph.

        Starts at top-level nodes and "zooms in" to children when relevant.
        """
        vector = self._get_embedding(query_text)

        knowledge_chunks = []
        self._recursive_search(vector, None, 0, max_depth, threshold, knowledge_chunks)

        return "\n\n".join(knowledge_chunks)

    async def query_async(self, query_text: str, max_depth: int = 2, threshold: float = 0.7) -> str:
        """Async version of query."""
        vector = await self._get_embedding_async(query_text)

        knowledge_chunks = []
        self._recursive_search(vector, None, 0, max_depth, threshold, knowledge_chunks)

        return "\n\n".join(knowledge_chunks)

    def _recursive_search(
        self,
        vector: List[float],
        parent_id: Optional[str],
        current_depth: int,
        max_depth: int,
        threshold: float,
        results_accumulator: List[str],
    ):
        """Depth-first recursive search through the knowledge hierarchy."""
        if current_depth > max_depth:
            return

        hits = self.store.search(vector, parent_id=parent_id, top_k=3)
        for hit in hits:
            if hit["score"] >= threshold:
                depth_marker = "  " * current_depth
                results_accumulator.append(
                    f"{depth_marker}[L{current_depth}] {hit['content'][:500]}"
                )
                # Zoom in: Search for children of this relevant node
                self._recursive_search(
                    vector, hit["id"], current_depth + 1, max_depth, threshold, results_accumulator
                )

    async def summarize_knowledge(
        self, query_text: str, max_depth: int = 2, threshold: float = 0.6
    ) -> str:
        """
        Retrieves and summarizes knowledge relevant to the query.

        Uses the LLM to create a cohesive summary from the retrieved chunks.
        """
        raw_knowledge = await self.query_async(query_text, max_depth, threshold)

        if not raw_knowledge:
            return ""

        summary = await self.llm.summarize(
            f"Summarize the following knowledge relevant to '{query_text}':\n\n{raw_knowledge}"
        )

        return summary
