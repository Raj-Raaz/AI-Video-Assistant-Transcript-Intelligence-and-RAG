from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda

import os

def get_llm():
    return ChatMistralAI(
        model="Mmistral-small-latest",
        mistral_api_key=os.environ.get("MISTRAL_API_KEY"),
        temperature=0.0,
        max_output_tokens=2048
    )


def split_text(text: str) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    return text_splitter.split_text(text)

def summarize_text(text: str) -> str:
    llm = get_llm()
    map_prompt = ChatPromptTemplate.from_template(
        [
            {"system": "You are a helpful assistant that summarizes text."},
            {"user": "Summarize the following text in a concise manner, highlighting the key points and main ideas. Ensure that the summary is clear and easy to understand.\n\nText:\n{text}\n\nSummary:"}
        ]
    )
    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_text(text)

    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]

    combined_summary = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_template(
        [(
            "system", "You are a helpful assistant that summarizes text."),
            ("user", "Combine the following summaries into a single, coherent summary that captures the main points and key ideas. Ensure that the final summary is clear, concise, and easy to understand.\n\nSummaries:\n{summaries}\n\nFinal Summary:")
       ]   
    )


    combined_chain = (
        RunnableLambda(lambda x: {"summaries": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    return combined_chain.invoke(combined_summary)

def generate_title(text: str) -> str:
    llm = get_llm()
    title_prompt = ChatPromptTemplate.from_template(
        [
            {"system": "You are a helpful assistant that generates titles for text."},
            {"user": "Generate a concise and descriptive title for the following text:\n\nText:\n{text}\n\nTitle:"}
        ]
    )
    title_chain = title_prompt | llm | StrOutputParser()
    return title_chain.invoke({"text": text})