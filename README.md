# youtube-rag-assistant

A RAG (Retrieval Augmented Generation) system that lets you ask questions about any YouTube video. It pulls the transcript, breaks it into chunks, stores them in a vector database (Qdrant), and then uses an LLM to answer your questions based on what was actually said in the video.

I built this while learning about RAG pipelines. The main idea was — instead of reading a 2 hour video, just ask it questions.

---

## What it does

- Takes a YouTube URL
- Fetches the transcript (English, falls back to Hindi if no English captions)
- Splits the transcript into overlapping chunks while keeping track of timestamps
- Embeds the chunks using `all-MiniLM-L6-v2` (384 dimensions)
- Stores everything in Qdrant vector DB (cloud or in-memory)
- When you ask a question, it finds the most relevant chunks and sends them to an LLM (via Groq)
- The answer includes clickable timestamp links so you can jump to that part in the video

---

## Stack

- **FastAPI** — backend API
- **Qdrant** — vector database for storing and searching embeddings
- **SentenceTransformers** — for generating embeddings (`all-MiniLM-L6-v2`)
- **Groq** — LLM inference (fast)
- **youtube-transcript-api** — to fetch transcripts

---

## Setup

You need Python and [uv](https://docs.astral.sh/uv/) installed.

```bash
git clone https://github.com/Jawad000000/youtube-rag-assistant.git
cd youtube-rag-assistant
uv sync
```

Copy `.env.example` to `.env` and fill in your keys:

```env
GROQ_API_KEY=your_groq_key_here
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_key_here
GROQ_MODEL=openai/gpt-oss-20b
COLLECTION_NAME=youtube_knowledge
DEFAULT_CHUNK_SIZE=500
DEFAULT_CHUNK_OVERLAP=60
```

If you leave the Qdrant credentials empty it'll just use an in-memory instance (good for testing).

---

## Running it

```bash
$env:PYTHONPATH="."
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` in your browser — there's a simple UI to test it out.

Swagger docs are at `http://localhost:8000/docs`.

---

## API

### Index a video
`POST /api/v1/index-video`
```json
{
  "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "chunk_size": 500,
  "chunk_overlap": 60
}
```

### Ask a question
`POST /api/v1/query`
```json
{
  "query": "what did they say about transformers?",
  "video_id": "VIDEO_ID",
  "top_k": 3
}
```

The response will have the answer + timestamp citations so you can verify what part of the video it came from.

---

## How the chunking works

YouTube transcripts come in really small segments (sometimes 2-3 words per line). Embedding those individually gives terrible results because there's no context. So I aggregate consecutive segments until I hit ~500 characters, and keep a small overlap between chunks so nothing gets cut off at a boundary.

Each chunk stores the `start_time` and `end_time` from the original transcript so we can generate YouTube timestamp links like `https://youtu.be/VIDEO_ID?t=135`.

---

## Tests

```bash
uv run pytest
```

---

## Project structure

```
youtube-rag-assistant/
├── app/
│   ├── core/
│   │   ├── transcript_extractor.py   # fetches youtube transcript
│   │   ├── chunker.py                # splits into chunks with timestamps
│   │   ├── embeddings.py             # sentence transformer wrapper
│   │   ├── vector_store.py           # qdrant operations
│   │   └── generator.py              # groq llm call
│   ├── api/
│   │   └── routes.py
│   ├── models/
│   │   └── schemas.py
│   ├── static/
│   │   └── index.html                # simple testing UI
│   ├── config.py
│   └── main.py
├── tests/
├── pyproject.toml
└── .env.example
```
