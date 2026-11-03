"""Shared LLM configuration, retries, and Mistral fallback."""
import os
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import httpx
from langchain_core.runnables import RunnableLambda

class RateLimitError(RuntimeError):
    pass

GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-small-latest"


def get_llm(json_mode=False):
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
    mistral_key = os.getenv("MISTRAL_API_KEY")

    def complete(prompt):
        roles = {"system": "system", "human": "user", "ai": "assistant"}
        messages = [
            {"role": roles[message.type], "content": message.content}
            for message in prompt.to_messages()
        ]

        def request(provider):
            is_groq = provider == "Groq"
            payload = {
                "model": GROQ_MODEL if is_groq else MISTRAL_MODEL,
                "messages": messages,
                "temperature": 0.0,
                "max_completion_tokens" if is_groq else "max_tokens": 2048,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            api_key = groq_key if is_groq else mistral_key
            response = httpx.post(
                GROQ_ENDPOINT if is_groq else MISTRAL_ENDPOINT,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError(f"{provider} returned an empty response. Try again.")
            return content

        request_chain = RunnableLambda(request)
        try:
            return invoke_with_retry(request_chain, "Groq", provider="Groq")
        except RateLimitError:
            if not mistral_key:
                raise RateLimitError(
                    "Groq is rate limited and MISTRAL_API_KEY is not configured for fallback."
                ) from None
            print("Groq is still rate limited. Trying Mistral for this request...")
            return invoke_with_retry(request_chain, "Mistral", provider="Mistral")

    return RunnableLambda(complete)


def retry_after_seconds(response):
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None

def invoke_with_retry(chain, inputs, max_attempts=4, provider="Groq"):
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    for attempt in range(max_attempts):
        try:
            return chain.invoke(inputs)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 429:
                raise
            delay = retry_after_seconds(exc.response)
            if attempt == max_attempts - 1 or (delay is not None and delay > 60):
                raise RateLimitError(
                    f"{provider} is still rate limiting requests. Try again later and "
                    f"check API usage/limits in the {provider} Console. "
                    "Retries cannot bypass a provider limit."
                ) from exc
            if delay is None:
                delay = 5 * (2 ** attempt) + random.uniform(0, 1)
            print(f"{provider} rate limit (429). Retrying in {delay:.1f} seconds "
                  f"({attempt + 2}/{max_attempts})...")
            time.sleep(delay)
