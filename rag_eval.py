import os
import json

from dotenv import load_dotenv
from groq import Groq

# need to set PYTHONPATH=. before running this
# uv run python rag_eval.py

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_model = "openai/gpt-oss-120b"

groq_client = Groq(api_key=GROQ_API_KEY)


# -------------------------------------------------------
# first we need to index a test video and then ask
# questions about it to evaluate the RAG pipeline
# -------------------------------------------------------

# import the core modules
from app.core.transcript_extractor import fetch_transcript
from app.core.chunker import chunk_transcript
from app.core.embeddings import embed_texts, embed_query
from app.core import vector_store
from app.core.generator import ask_llm

# test video - this is a python tutorial video for easy questions
TEST_VIDEO_URL = "https://www.youtube.com/watch?v=bSrm9RXwBaI"

print("indexing test video...")

video_id, raw_segments = fetch_transcript(TEST_VIDEO_URL)

print(f"got {len(raw_segments)} segments")

chunks = chunk_transcript(video_id, raw_segments, chunk_size=500, chunk_overlap=60)

print(f"created {len(chunks)} chunks")

texts = [c["text"] for c in chunks]
vectors = embed_texts(texts)

vector_store.upsert_chunks(chunks, vectors)

print("uploaded to qdrant!")
print()


# -------------------------------------------------------
# search function - same as fullq.py
# -------------------------------------------------------

def search(query, top_k=3):
    query_vector = embed_query(query)
    results = vector_store.search(query_vector, top_k=top_k, video_id=video_id)
    return results


# -------------------------------------------------------
# llm as a judge - ask the llm to score things
# returns json
# -------------------------------------------------------

def llm_judge(prompt):
    response = groq_client.chat.completions.create(
        model=groq_model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    return json.loads(response.choices[0].message.content)


# -------------------------------------------------------
# metric 1 - context precision
# for each retrieved chunk, is it actually relevant?
# precision = relevant chunks / total chunks retrieved
# -------------------------------------------------------

def context_precision(question, retrieved_chunks):
    relevant = 0

    for chunk in retrieved_chunks:
        text = chunk["text"]

        prompt = f"""You are checking if a retrieved chunk is useful for answering a question.

Question: {question}

Retrieved chunk: {text}

Is this chunk relevant to the question?

Return ONLY JSON:
{{
    "relevant": true,
    "reason": "short explanation"
}}

true if it helps answer the question, false if its unrelated."""

        result = llm_judge(prompt)

        if result["relevant"]:
            relevant += 1

    if len(retrieved_chunks) == 0:
        return 0.0

    return relevant / len(retrieved_chunks)


# -------------------------------------------------------
# metric 2 - context recall
# did we retrieve the info needed to answer?
# -------------------------------------------------------

def context_recall(question, context, ground_truth):
    prompt = f"""You are evaluating a RAG system.

Question: {question}

Ground Truth Answer: {ground_truth}

Retrieved Context: {context}

Does the retrieved context have enough info to produce the ground truth answer?

Return ONLY JSON:
{{
    "score": 0.0,
    "reason": "short explanation"
}}

1.0 = all the info is there
0.7 = most info is there, minor stuff missing
0.5 = some info is there
0.0 = the info is not there at all"""

    return llm_judge(prompt)


# -------------------------------------------------------
# metric 3 - faithfulness
# is the answer actually supported by the context?
# or is the llm hallucinating?
# -------------------------------------------------------

def evaluate_faithfulness(question, context, answer):
    prompt = f"""You are evaluating a RAG system.

Question: {question}

Retrieved Context: {context}

Generated Answer: {answer}

Are the claims in the answer supported by the context?

Return ONLY JSON:
{{
    "score": 0.0,
    "reason": "short explanation"
}}

1.0 = everything is supported
0.7 = mostly supported, minor issues
0.5 = some claims are supported
0.0 = answer is not supported or contradicts context"""

    return llm_judge(prompt)


# -------------------------------------------------------
# metric 4 - answer relevancy
# does the answer actually answer the question?
# -------------------------------------------------------

def evaluate_relevancy(question, answer):
    prompt = f"""You are evaluating a RAG system.

Question: {question}

Generated Answer: {answer}

Does the answer actually answer the question?

Return ONLY JSON:
{{
    "score": 0.0,
    "reason": "short explanation"
}}

1.0 = directly answers the question
0.7 = mostly answers it
0.5 = partially answers it
0.0 = completely off topic"""

    return llm_judge(prompt)


# -------------------------------------------------------
# metric 5 - answer correctness
# is the answer correct compared to ground truth?
# -------------------------------------------------------

def evaluate_correctness(answer, ground_truth):
    prompt = f"""You are evaluating a RAG system.

Generated Answer: {answer}

Ground Truth Answer: {ground_truth}

Is the generated answer factually correct?

Return ONLY JSON:
{{
    "score": 0.0,
    "reason": "short explanation"
}}

1.0 = completely correct
0.7 = mostly correct, minor omissions
0.5 = partially correct
0.0 = incorrect or contradicts ground truth"""

    return llm_judge(prompt)


# -------------------------------------------------------
# golden dataset - questions we know the answers to
# about the test video
# change these if you use a different video
# -------------------------------------------------------

# NOTE: these are generic questions that should work
# for any tutorial video - update ground_truth if needed
golden_dataset = [
    {
        "id": 1,
        "question": "what is the main topic of this video?",
        "ground_truth": "the video is a tutorial covering python programming concepts"
    },
    {
        "id": 2,
        "question": "what concepts are explained in this video?",
        "ground_truth": "the video explains python programming concepts and examples"
    },
    {
        "id": 3,
        "question": "what programming language is used in this tutorial?",
        "ground_truth": "python"
    }
]


# -------------------------------------------------------
# run the full evaluation
# -------------------------------------------------------

def run_evaluation():

    print()
    print("=" * 60)
    print("           RAG EVALUATION")
    print("=" * 60)

    all_scores = {
        "precision": [],
        "recall": [],
        "faithfulness": [],
        "relevancy": [],
        "correctness": []
    }

    for test in golden_dataset:
        question = test["question"]
        ground_truth = test["ground_truth"]

        print()
        print("-" * 60)
        print(f"QUESTION: {question}")

        # retrieval
        results = search(question, top_k=3)

        print("\nRetrieved chunks:")
        for i, r in enumerate(results):
            print(f"\nChunk {i+1}")
            print(f"Score: {r['score']}")
            print(f"Timestamp: {r['timestamp_formatted']}")
            print(r["text"][:200] + "...")  # just show first 200 chars

        # build context string
        context = "\n".join(r["text"] for r in results)

        # generate answer
        answer = ask_llm(question, results)

        print(f"\nAnswer: {answer}")

        # score it
        precision = context_precision(question, results)

        recall_result = context_recall(question, context, ground_truth)
        recall = float(recall_result["score"])

        faith_result = evaluate_faithfulness(question, context, answer)
        faithfulness = float(faith_result["score"])

        rel_result = evaluate_relevancy(question, answer)
        relevancy = float(rel_result["score"])

        correct_result = evaluate_correctness(answer, ground_truth)
        correctness = float(correct_result["score"])

        all_scores["precision"].append(precision)
        all_scores["recall"].append(recall)
        all_scores["faithfulness"].append(faithfulness)
        all_scores["relevancy"].append(relevancy)
        all_scores["correctness"].append(correctness)

        print("\nScores:")
        print(f"  Context Precision : {precision:.2f}")
        print(f"  Context Recall    : {recall:.2f}")
        print(f"  Faithfulness      : {faithfulness:.2f}")
        print(f"  Answer Relevancy  : {relevancy:.2f}")
        print(f"  Answer Correctness: {correctness:.2f}")

        # quick diagnosis
        print("\nDiagnosis:")
        if precision < 0.7:
            print("  bad retrieval - returning irrelevant chunks")
        if recall < 0.7:
            print("  missing info - retrieval not finding what it needs")
        if faithfulness < 0.7:
            print("  hallucination - answer not grounded in context")
        if relevancy < 0.7:
            print("  off topic - answer doesnt address the question")
        if correctness < 0.7:
            print("  wrong answer - doesnt match ground truth")

        if all(s >= 0.7 for s in [precision, recall, faithfulness, relevancy, correctness]):
            print("  looks good!")

    # final averages
    print()
    print("=" * 60)
    print("           FINAL SUMMARY")
    print("=" * 60)

    for metric, values in all_scores.items():
        avg = sum(values) / len(values)
        print(f"{metric.capitalize():20}: {avg:.2f}")


if __name__ == "__main__":
    run_evaluation()
