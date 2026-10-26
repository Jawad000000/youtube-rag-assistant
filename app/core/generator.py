from groq import Groq
from app.config import settings

groq_client = Groq(api_key=settings.GROQ_API_KEY)
groq_model = settings.GROQ_MODEL


def ask_llm(query, context_chunks):
    if not context_chunks:
        return "i couldnt find anything relevant in the video for this question"

    # build context string with timestamps
    context = ""
    for i, chunk in enumerate(context_chunks, 1):
        ts = chunk["timestamp_formatted"]
        context += f"[{i}] at {ts}:\n{chunk['text'].strip()}\n\n"

    prompt = f"""Answer the question using only what is said in the video transcript below.
If the answer is not in the transcript, say: "this video doesnt cover that topic"

Video Transcript:
{context}

Question: {query}

Answer:"""

    response = groq_client.chat.completions.create(
        model=groq_model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=600
    )

    return response.choices[0].message.content.strip()
