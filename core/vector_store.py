
import os
from typing import Any

from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

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
    return QdrantClient(url="http://localhost:6333")


def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings(),
    )


def index_transcript(
    transcript: str,
    video_id: str,
    metadata: dict[str, Any] | None = None,
) -> int:

    if not transcript.strip():
        raise ValueError("Transcript cannot be empty")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    documents = splitter.create_documents(
        texts=[transcript],
        metadatas=[metadata or {}],
    )

    # Add video information to every chunk
    for i, doc in enumerate(documents):
        doc.metadata["video_id"] = video_id
        doc.metadata["chunk_index"] = i

    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        client=get_qdrant_client(),
    )

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
            filter={
                "must": [
                    {
                        "key": "metadata.video_id",
                        "match": {"value": video_id},
                    }
                ]
            },
        )

    return vector_store.similarity_search(query, k=k)
