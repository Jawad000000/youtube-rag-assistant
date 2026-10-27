import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType
)
from app.config import settings

# connect to qdrant - cloud if keys are set, otherwise in-memory
if settings.QDRANT_URL and settings.QDRANT_API_KEY:
    try:
        qdrant = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=10.0
        )
        qdrant.get_collections()  # test the connection
        print("connected to qdrant cloud!")
    except Exception as e:
        print(f"qdrant cloud failed, using in-memory: {e}")
        qdrant = QdrantClient(":memory:")
else:
    print("no qdrant credentials, using in-memory")
    qdrant = QdrantClient(":memory:")

COLLECTION_NAME = settings.COLLECTION_NAME
EMBEDDING_SIZE = settings.EMBEDDING_SIZE

def init_collection():
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_SIZE,
                distance=Distance.COSINE
            )
        )
        try:
            qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="video_id",
                field_schema=PayloadSchemaType.KEYWORD
            )
        except Exception:
            pass
        print(f"created collection: {COLLECTION_NAME}")

init_collection()

def clear_collection():
    """Wipes all previous video vectors completely to prevent cross-video pollution."""
    try:
        if qdrant.collection_exists(COLLECTION_NAME):
            qdrant.delete_collection(COLLECTION_NAME)
            print(f"deleted old collection: {COLLECTION_NAME}")
        init_collection()
    except Exception as e:
        print(f"error clearing collection: {e}")


def upsert_chunks(chunks, vectors):
    if not chunks or not vectors:
        return

    points = []
    for chunk, vector in zip(chunks, vectors):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))
        points.append(PointStruct(
            id=point_id,
            vector=vector,
            payload=chunk
        ))

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"upserted {len(points)} points to qdrant")


def search(query_vector, top_k=3, video_id=None):
    search_filter = None
    if video_id:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="video_id",
                    match=MatchValue(value=video_id)
                )
            ]
        )

    try:
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=search_filter,
            limit=top_k
        ).points
    except AttributeError:
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=top_k
        )

    chunks = []
    for res in results:
        p = res.payload or {}
        chunks.append({
            "chunk_id": p.get("chunk_id", ""),
            "text": p.get("text", ""),
            "video_id": p.get("video_id", ""),
            "start_seconds": p.get("start_seconds", 0.0),
            "end_seconds": p.get("end_seconds", 0.0),
            "timestamp_formatted": p.get("timestamp_formatted", ""),
            "youtube_timestamp_url": p.get("youtube_timestamp_url", ""),
            "score": round(float(res.score), 4) if hasattr(res, "score") else None
        })

    return chunks
