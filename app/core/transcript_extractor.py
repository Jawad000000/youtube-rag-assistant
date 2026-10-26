import re
from youtube_transcript_api import YouTubeTranscriptApi


# extract the video id from whatever url the user gives
def extract_video_id(url):
    url = url.strip()

    # if its already just the id (11 chars) return it directly
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url):
        return url

    # try common youtube url patterns
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:[&?]|$)",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"youtube\.com\/embed\/([0-9A-Za-z_-]{11})",
        r"youtube\.com\/shorts\/([0-9A-Za-z_-]{11})"
    ]

    for p in patterns:
        match = re.search(p, url)
        if match:
            return match.group(1)

    raise ValueError(f"cant find video id in: {url}")


# convert seconds to mm:ss format
def format_timestamp(seconds):
    s = int(seconds)
    hours = s // 3600
    minutes = (s % 3600) // 60
    secs = s % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def fetch_transcript(video_url):
    video_id = extract_video_id(video_url)

    api = YouTubeTranscriptApi()

    # try english first, then hindi, then whatever is available
    try:
        transcript_raw = api.fetch(video_id, languages=["en", "en-US", "en-GB", "hi", "hi-Latn"])
    except Exception:
        try:
            # fallback - just grab whatever transcript exists
            t_list = api.list(video_id)
            transcript_raw = None
            for t in t_list:
                transcript_raw = t.fetch()
                break
        except Exception as e:
            raise RuntimeError(f"couldnt get transcript for {video_id}: {e}")

    if not transcript_raw:
        raise RuntimeError(f"no captions found for {video_id}")

    # clean up the segments into simple dicts
    segments = []
    for s in transcript_raw:
        text = s.text if hasattr(s, "text") else s.get("text", "")
        start = s.start if hasattr(s, "start") else s.get("start", 0.0)
        duration = s.duration if hasattr(s, "duration") else s.get("duration", 0.0)

        text = text.replace("\n", " ").strip()
        if text:
            segments.append({
                "text": text,
                "start": float(start),
                "duration": float(duration)
            })

    print(f"got {len(segments)} transcript segments")
    return video_id, segments
