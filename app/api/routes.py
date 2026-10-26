import time
from fastapi import APIRouter, HTTPException, status
from app.models.schemas import (
    VideoIndexRequest,
    VideoIndexResponse,
    QueryRequest,
    QueryResponse,
    HealthResponse
)
from app.core.transcript_extractor import fetch_transcript
from app.core.chunker import chunk_transcript
from app.core.embeddings import embed_texts, embed_query
from app.core import vector_store
from app.core.generator import ask_llm
from app.config import settings

router = APIRouter()


@router.post("/index-video", response_model=VideoIndexResponse, status_code=status.HTTP_201_CREATED)
async def index_youtube_video(payload: VideoIndexRequest):
    try:
        # step 1: get transcript
        video_id, raw_segments = fetch_transcript(payload.video_url)

        if not raw_segments:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="no captions found for this video"
            )

        # step 2: chunk the transcript
        chunk_size = payload.chunk_size or settings.DEFAULT_CHUNK_SIZE
        chunk_overlap = payload.chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP
        chunks = chunk_transcript(video_id, raw_segments, chunk_size, chunk_overlap)

        # step 3: embed the chunks
        texts = [c["text"] for c in chunks]
        vectors = embed_texts(texts)

        # step 4: store in qdrant
        vector_store.upsert_chunks(chunks, vectors)

        last_seg = raw_segments[-1]
        duration = last_seg.get("start", 0.0) + last_seg.get("duration", 0.0)

        return VideoIndexResponse(
            video_id=video_id,
            total_segments=len(raw_segments),
            total_chunks_created=len(chunks),
            total_duration_seconds=round(duration, 2),
            message=f"indexed {len(chunks)} chunks for video {video_id}"
        )

    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"indexing failed: {e}")


@router.post("/query", response_model=QueryResponse)
async def query_video(payload: QueryRequest):
    t0 = time.perf_counter()

    # embed the query
    query_vector = embed_query(payload.query)

    # search qdrant
    top_chunks = vector_store.search(
        query_vector=query_vector,
        top_k=payload.top_k or settings.DEFAULT_TOP_K,
        video_id=payload.video_id
    )

    t1 = time.perf_counter()
    retrieval_ms = (t1 - t0) * 1000

    # ask the llm
    answer = ask_llm(payload.query, top_chunks)

    t2 = time.perf_counter()
    gen_ms = (t2 - t1) * 1000
    total_ms = (t2 - t0) * 1000

    # build citations in the schema format
    from app.models.schemas import VideoChunkDetail, VideoChunkMetadata
    citations = []
    for c in top_chunks:
        meta = VideoChunkMetadata(
            video_id=c["video_id"],
            chunk_id=c["chunk_id"],
            start_seconds=c["start_seconds"],
            end_seconds=c["end_seconds"],
            timestamp_formatted=c["timestamp_formatted"],
            youtube_timestamp_url=c["youtube_timestamp_url"]
        )
        citations.append(VideoChunkDetail(
            chunk_id=c["chunk_id"],
            text=c["text"],
            metadata=meta,
            score=c.get("score")
        ))

    return QueryResponse(
        query=payload.query,
        answer=answer,
        video_id=payload.video_id,
        citations=citations,
        retrieval_latency_ms=round(retrieval_ms, 2),
        generation_latency_ms=round(gen_ms, 2),
        total_latency_ms=round(total_ms, 2)
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        qdrant_connected=True,
        groq_configured=bool(settings.GROQ_API_KEY),
        embedding_model=settings.EMBEDDING_MODEL_NAME
    )
