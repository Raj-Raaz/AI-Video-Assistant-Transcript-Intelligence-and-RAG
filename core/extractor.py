
import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnablePassthrough

def get_llm():
    return ChatMistralAI(
        model="Mmistral-small-latest",
        mistral_api_key=os.environ.get("MISTRAL_API_KEY"),
        temperature=0.2,
        max_output_tokens=2048
    )

def build_chain(system_prompt:str):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input_text}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain   