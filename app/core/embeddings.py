from sentence_transformers import SentenceTransformer
from app.config import settings

# load the model once - dont want to reload it every request
print("loading embedding model...")
model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
print("embedding model ready!")


def embed_texts(texts):
    if not texts:
        return []
    # normalize=True makes cosine similarity work better
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query):
    embedding = model.encode(query, normalize_embeddings=True)
    return embedding.tolist()
