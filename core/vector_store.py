
import os
from uuid import NAMESPACE_URL, uuid5
from typing import Any

from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

load_dotenv()

COLLECTION_NAME = "video_transcripts"
EMBEDDING_MODEL = "embed-english-v3.0"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5


def get_embeddings() -> CohereEmbeddings:
    api_key = os.getenv("COHERE_API_KEY")

    if not api_key:
        raise RuntimeError("COHERE_API_KEY is not set")

    return CohereEmbeddings(
        model=EMBEDDING_MODEL,
        cohere_api_key=api_key,
    )


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )


def get_vector_store() -> QdrantVectorStore:
    client = get_qdrant_client()
    embeddings = get_embeddings()
    if not client.collection_exists(COLLECTION_NAME):
        # Derive the vector size from the configured embedding model.
        dimension = len(embeddings.embed_query("embedding dimension"))
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )


def index_transcript(
    transcript: str,
    video_id: str,
    metadata: dict[str, Any] | None = None,
    vector_store: QdrantVectorStore | None = None,
    segments: list[dict] | None = None,
) -> int:

    if not transcript.strip():
        raise ValueError("Transcript cannot be empty")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    documents = splitter.create_documents(
        texts=[transcript],
        metadatas=[metadata or {}],
    )

    # Map character ranges back to the speech segments they overlap.
    ranges = []
    cursor = 0
    for segment in segments or []:
        start = transcript.find(segment["text"], cursor)
        if start < 0:
            raise ValueError("Timed segment text does not match the transcript")
        end = start + len(segment["text"])
        ranges.append((start, end, segment))
        cursor = end

    # Add video information to every chunk
    for i, doc in enumerate(documents):
        doc.metadata["video_id"] = video_id
        doc.metadata["chunk_index"] = i
        start = doc.metadata["start_index"]
        end = start + len(doc.page_content)
        matches = [s for left, right, s in ranges if left < end and right > start]
        if matches:
            doc.metadata["start_seconds"] = matches[0]["start"]
            doc.metadata["end_seconds"] = matches[-1]["end"]

    if vector_store is None:
        vector_store = get_vector_store()
    # Upsert the same IDs on repeat runs instead of creating duplicate chunks.
    ids = [str(uuid5(NAMESPACE_URL, f"{video_id}:{i}")) for i in range(len(documents))]
    vector_store.add_documents(documents, ids=ids)

    return len(documents)


def search_transcript(
    query: str,
    k: int = TOP_K,
    video_id: str | None = None,
) -> list[Document]:

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if k < 1:
        raise ValueError("k must be greater than zero")

    vector_store = get_vector_store()

    if video_id:
        return vector_store.similarity_search(
            query=query,
            k=k,
            filter=video_filter(video_id),
        )

    return vector_store.similarity_search(query, k=k)


def video_filter(video_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(
        key="metadata.video_id", match=models.MatchValue(value=video_id),
    )])
