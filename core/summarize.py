from core.llm_utils import get_llm, invoke_with_retry
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(text: str) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    return text_splitter.split_text(text)

def summarize_text(text: str) -> str:
    llm = get_llm()
    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that summarizes text."),
            ("user", "Summarize the following text in a concise manner, highlighting the key points and main ideas. Ensure that the summary is clear and easy to understand.\n\nText:\n{text}\n\nSummary:")
        ]
    )
    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_text(text)

    chunk_summaries = [invoke_with_retry(map_chain, {"text": chunk}) for chunk in chunks]

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    combined_summary = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that summarizes text."),
            ("user", "Combine the following summaries into a single, coherent summary that captures the main points and key ideas. Ensure that the final summary is clear, concise, and easy to understand.\n\nSummaries:\n{summaries}\n\nFinal Summary:")
        ]
    )


    combined_chain = combined_prompt | llm | StrOutputParser()
    return invoke_with_retry(combined_chain, {"summaries": combined_summary})


def generate_title(text: str) -> str:
    llm = get_llm()
    title_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that generates titles for text."),
            ("user", "Generate a concise and descriptive title for the following text:\n\nText:\n{text}\n\nTitle:")
        ]
    )
    title_chain = title_prompt | llm | StrOutputParser()
    return invoke_with_retry(title_chain, {"text": text})