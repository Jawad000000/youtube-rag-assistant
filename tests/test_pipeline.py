import pytest
from app.core.transcript_extractor import extract_video_id, format_timestamp
from app.core.chunker import chunk_transcript


def test_extract_video_id():
    url1 = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    url2 = "https://youtu.be/dQw4w9WgXcQ"
    url3 = "dQw4w9WgXcQ"

    assert extract_video_id(url1) == "dQw4w9WgXcQ"
    assert extract_video_id(url2) == "dQw4w9WgXcQ"
    assert extract_video_id(url3) == "dQw4w9WgXcQ"


def test_format_timestamp():
    assert format_timestamp(45) == "00:45"
    assert format_timestamp(125) == "02:05"
    assert format_timestamp(3665) == "01:01:05"


def test_chunker():
    dummy_segments = [
        {"text": "Hello world welcome to this tutorial.", "start": 0.0, "duration": 3.0},
        {"text": "Today we will talk about fast APIs and vector databases.", "start": 3.0, "duration": 4.0},
        {"text": "Let us get started right away.", "start": 7.0, "duration": 2.5}
    ]

    chunks = chunk_transcript("test_vid_123", dummy_segments, chunk_size=50, chunk_overlap=10)

    assert len(chunks) >= 1
    assert chunks[0]["video_id"] == "test_vid_123"
    assert "https://www.youtube.com/watch?v=test_vid_123&t=" in chunks[0]["youtube_timestamp_url"]
    assert chunks[0]["start_seconds"] == 0.0
