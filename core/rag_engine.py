import hashlib
import json
import math
import re
from urllib.parse import parse_qs, urlparse
from dotenv import load_dotenv

from core.llm_utils import get_llm, invoke_with_retry
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from core.vector_store import get_vector_store, index_transcript, video_filter

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def timestamp(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def youtube_url(source):
    parsed = urlparse(source or "")
    host = (parsed.hostname or "").lower()
    parts = parsed.path.strip("/").split("/")
    video_id = None
    if parsed.scheme not in ("http", "https"):
        return None
    if host == "youtu.be":
        video_id = parts[0]
    elif host == "youtube.com" or host.endswith(".youtube.com"):
        video_id = parse_qs(parsed.query).get("v", [None])[0]
        if not video_id and len(parts) == 2 and parts[0] in ("shorts", "embed", "live"):
            video_id = parts[1]
    if video_id and re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        return f"https://www.youtube.com/watch?v={video_id}"
    return None


def source_citations(docs, source):
    lines = []
    seen = set()
    url = youtube_url(source)
    for doc in docs:
        start = doc.metadata.get("start_seconds")
        end = doc.metadata.get("end_seconds")
        if start is None or end is None or (start, end) in seen:
            continue
        seen.add((start, end))
        label = f"{timestamp(start)}–{timestamp(math.ceil(end))}"
        lines.append(f"- [{label}]({url}&t={int(start)}s)" if url else f"- {label}")
    return "\n\n**Retrieved sources**\n" + "\n".join(lines) if lines else ""


def build_rag_chain(transcript: str, segments=None, source=None):

    if not transcript.strip():
        raise ValueError("Transcript cannot be empty")
    # Keep timed indexes separate from older untimed chunks and other sources.
    identity = json.dumps([transcript, segments, youtube_url(source)], sort_keys=True)
    video_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    vector_store = get_vector_store()
    metadata = {"source_url": youtube_url(source)} if youtube_url(source) else {}
    count = index_transcript(transcript, video_id, metadata=metadata,
                             vector_store=vector_store, segments=segments)
    print(f"Indexed {count} transcript chunks for video chat.")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5, "filter": video_filter(video_id)},
    )

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert AI video assistant.

Answer the user's question based ONLY on the video transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the video transcripts."

Always be concise and precise.
If quoting someone, mention it clearly.
Do not invent timestamps or source links; the application displays retrieved sources separately.

Context from video transcripts:
{context}""",
        ),
        ("human", "{question}"),
    ])

    answer_chain = prompt | llm | StrOutputParser()

    def answer(question):
        docs = retriever.invoke(question)
        response = answer_chain.invoke({"context": format_docs(docs), "question": question})
        return response + source_citations(docs, source)

    return RunnableLambda(answer)


def ask_question(rag_chain, question: str) -> str:

    if not question.strip():
        raise ValueError("Question cannot be empty")

    return invoke_with_retry(rag_chain, question)
