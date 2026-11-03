import json

from langchain_core.prompts import ChatPromptTemplate

from core.llm_utils import get_llm


def extract_information(text: str) -> dict[str, str]:
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Extract information explicitly stated in the transcript. "
         "Return a JSON object with exactly three keys: action_items, key_decisions, questions. "
         "Each value must be a string containing a numbered list. "
         "For action_items, include only assigned tasks or explicit commitments, with owner "
         "and deadline (use 'Not specified' when absent). Do not turn reported events into tasks. "
         "For key_decisions, include only explicitly stated decisions. "
         "For questions, include only unresolved questions actually raised in the transcript; "
         "do not invent follow-up questions. "
         "If a category is absent, use 'No action items found.', 'No key decisions found.', "
         "or 'No open questions found.' respectively. Do not guess names, dates, or facts."),
        ("human", "{text}"),
    ])
    chain = prompt | get_llm(json_mode=True)
    response = chain.invoke({"text": text})
    try:
        result = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError("The model returned invalid JSON for the extracted information.") from exc

    fields = ("action_items", "key_decisions", "questions")
    if not isinstance(result, dict) or any(
        not isinstance(result.get(field), str) or not result[field].strip()
        for field in fields
    ):
        raise ValueError("The model must return text for all three extraction fields.")
    return {field: result[field] for field in fields}
