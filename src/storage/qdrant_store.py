from __future__ import annotations

from dataclasses import dataclass
import uuid

import numpy as np

from ..schemas import PageRecord


@dataclass(frozen=True)
class QdrantVisualConfig:
    collection_name: str = "visual_pages_multivector"
    url: str | None = None
    api_key: str | None = None
    local_path: str | None = None


@dataclass(frozen=True)
class QdrantSingleVectorConfig:
    collection_name: str = "visual_pages_proxy"
    url: str | None = None
    api_key: str | None = None
    local_path: str | None = None


class QdrantVisualStore:
    def __init__(self, config: QdrantVisualConfig):
        from qdrant_client import QdrantClient

        self.config = config
        if config.url:
            self.client = QdrantClient(url=config.url, api_key=config.api_key, timeout=120)
        elif config.local_path:
            self.client = QdrantClient(path=config.local_path)
        else:
            self.client = QdrantClient(":memory:")

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        from qdrant_client.http.models import Distance, MultiVectorComparator, MultiVectorConfig, VectorParams

        if recreate:
            self.client.recreate_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
                ),
            )
            return

        existing = {collection.name for collection in self.client.get_collections().collections}
        if self.config.collection_name in existing:
            return
        self.client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
            ),
        )

    def upsert_pages(self, page_records: list[PageRecord], page_vectors: list[np.ndarray]) -> None:
        from qdrant_client.http.models import PointStruct

        if not page_records or not page_vectors:
            return
        vector_size = int(page_vectors[0].shape[-1])
        self.ensure_collection(vector_size)
        points = [
            PointStruct(
                id=_stable_point_id(page.page_id),
                vector=vector.astype("float32").tolist(),
                payload=page.payload(),
            )
            for page, vector in zip(page_records, page_vectors)
        ]
        self.client.upsert(
            collection_name=self.config.collection_name,
            points=points,
            wait=True,
        )

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        response = self.client.query_points(
            collection_name=self.config.collection_name,
            query=query_vector.astype("float32").tolist(),
            limit=max(1, top_k),
            with_payload=True,
            with_vectors=False,
        )
        return getattr(response, "points", response)

    def count(self) -> int:
        count = self.client.count(collection_name=self.config.collection_name, exact=True)
        return int(getattr(count, "count", 0))

    def status(self) -> dict:
        return {
            "collection_name": self.config.collection_name,
            "point_count": self.count(),
        }


class QdrantSingleVectorStore:
    def __init__(self, config: QdrantSingleVectorConfig):
        from qdrant_client import QdrantClient

        self.config = config
        if config.url:
            self.client = QdrantClient(url=config.url, api_key=config.api_key, timeout=120)
        elif config.local_path:
            self.client = QdrantClient(path=config.local_path)
        else:
            self.client = QdrantClient(":memory:")

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        from qdrant_client.http.models import Distance, VectorParams

        if recreate:
            self.client.recreate_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            return

        existing = {collection.name for collection in self.client.get_collections().collections}
        if self.config.collection_name in existing:
            return
        self.client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def upsert_pages(self, page_records: list[PageRecord], page_vectors: list[np.ndarray]) -> None:
        from qdrant_client.http.models import PointStruct

        if not page_records or not page_vectors:
            return
        vector_size = int(page_vectors[0].shape[-1])
        self.ensure_collection(vector_size)
        points = [
            PointStruct(
                id=_stable_point_id(page.page_id),
                vector=vector.astype("float32").tolist(),
                payload=page.payload(),
            )
            for page, vector in zip(page_records, page_vectors)
        ]
        self.client.upsert(
            collection_name=self.config.collection_name,
            points=points,
            wait=True,
        )

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        response = self.client.query_points(
            collection_name=self.config.collection_name,
            query=query_vector.astype("float32").tolist(),
            limit=max(1, top_k),
            with_payload=True,
            with_vectors=False,
        )
        return getattr(response, "points", response)

    def count(self) -> int:
        count = self.client.count(collection_name=self.config.collection_name, exact=True)
        return int(getattr(count, "count", 0))

    def status(self) -> dict:
        return {
            "collection_name": self.config.collection_name,
            "point_count": self.count(),
        }


def _stable_point_id(page_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, page_id))
