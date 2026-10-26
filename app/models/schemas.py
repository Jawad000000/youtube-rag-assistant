from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any

class VideoIndexRequest(BaseModel):
    video_url: str = Field(..., description="YouTube video URL or video ID (e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ)")
    chunk_size: Optional[int] = Field(500, ge=100, le=2000, description="Target character length for each transcript chunk.")
    chunk_overlap: Optional[int] = Field(60, ge=0, le=500, description="Overlapping character length between consecutive chunks.")

class VideoChunkMetadata(BaseModel):
    video_id: str
    chunk_id: str
    start_seconds: float
    end_seconds: float
    timestamp_formatted: str
    youtube_timestamp_url: str

class VideoChunkDetail(BaseModel):
    chunk_id: str
    text: str
    metadata: VideoChunkMetadata
    score: Optional[float] = None

class VideoIndexResponse(BaseModel):
    video_id: str
    title: Optional[str] = "YouTube Video"
    total_segments: int
    total_chunks_created: int
    total_duration_seconds: float
    message: str

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Question about the video content.")
    video_id: Optional[str] = Field(None, description="Optional YouTube video ID to filter the search. If omitted, searches across all indexed videos.")
    top_k: Optional[int] = Field(3, ge=1, le=10, description="Number of context segments to retrieve.")

class QueryResponse(BaseModel):
    query: str
    answer: str
    video_id: Optional[str] = None
    citations: List[VideoChunkDetail]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float

class HealthResponse(BaseModel):
    status: str
    version: str
    qdrant_connected: bool
    groq_configured: bool
    embedding_model: str
