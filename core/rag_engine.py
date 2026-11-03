import hashlib
from dotenv import load_dotenv

from core.llm_utils import get_llm, invoke_with_retry
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import get_vector_store, index_transcript, video_filter

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(transcript: str):

    if not transcript.strip():
        raise ValueError("Transcript cannot be empty")
    video_id = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    vector_store = get_vector_store()
    count = index_transcript(transcript, video_id, vector_store=vector_store)
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

Context from video transcripts:
{context}""",
        ),
        ("human", "{question}"),
    ])

    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def ask_question(rag_chain, question: str) -> str:

    if not question.strip():
        raise ValueError("Question cannot be empty")

    return invoke_with_retry(rag_chain, question)
