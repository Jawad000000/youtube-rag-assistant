import uuid
from app.core.transcript_extractor import format_timestamp


# chunk the raw transcript segments into bigger pieces
# youtube captions are like 2-3 words each so embedding them directly is bad
# we join them into ~500 char chunks and track timestamps
def chunk_transcript(video_id, raw_segments, chunk_size=500, chunk_overlap=60):
    if not raw_segments:
        return []

    chunks = []
    current_parts = []
    current_chars = 0
    current_start = raw_segments[0].get("start", 0.0)
    current_end = current_start
    chunk_idx = 0

    for segment in raw_segments:
        seg_text = segment.get("text", "").strip()
        seg_start = segment.get("start", 0.0)
        seg_dur = segment.get("duration", 0.0)
        seg_end = seg_start + seg_dur

        if not seg_text:
            continue

        current_parts.append(seg_text)
        current_chars += len(seg_text) + 1
        current_end = max(current_end, seg_end)

        if current_chars >= chunk_size:
            full_text = " ".join(current_parts)
            ts_formatted = f"{format_timestamp(current_start)} - {format_timestamp(current_end)}"
            yt_url = f"https://www.youtube.com/watch?v={video_id}&t={int(current_start)}s"
            chunk_id = f"{video_id}_{chunk_idx}_{uuid.uuid4().hex[:6]}"

            chunks.append({
                "chunk_id": chunk_id,
                "text": full_text,
                "video_id": video_id,
                "start_seconds": round(current_start, 2),
                "end_seconds": round(current_end, 2),
                "timestamp_formatted": ts_formatted,
                "youtube_timestamp_url": yt_url
            })

            chunk_idx += 1

            # keep last segment for overlap so we dont lose context at boundaries
            if len(current_parts) > 1 and chunk_overlap > 0:
                current_parts = current_parts[-1:]
                current_chars = sum(len(p) + 1 for p in current_parts)
                current_start = seg_start
            else:
                current_parts = []
                current_chars = 0
                current_start = seg_end

    # dont forget the last remaining bits
    if current_parts:
        full_text = " ".join(current_parts)
        ts_formatted = f"{format_timestamp(current_start)} - {format_timestamp(current_end)}"
        yt_url = f"https://www.youtube.com/watch?v={video_id}&t={int(current_start)}s"
        chunk_id = f"{video_id}_{chunk_idx}_{uuid.uuid4().hex[:6]}"

        chunks.append({
            "chunk_id": chunk_id,
            "text": full_text,
            "video_id": video_id,
            "start_seconds": round(current_start, 2),
            "end_seconds": round(current_end, 2),
            "timestamp_formatted": ts_formatted,
            "youtube_timestamp_url": yt_url
        })

    print(f"created {len(chunks)} chunks")
    return chunks
