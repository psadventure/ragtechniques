import os
import re
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic

from config import settings
from generation.prompt_builder import PromptMessages

load_dotenv()

CITATION_PATTERN = r'\[Source: .+?, Section: .+?\]'


@dataclass
class Answer:
    text: str
    citations: List[str]   # extracted citation strings from the answer text
    query: str
    chunks_used: int


def generateUsingOpenAI(prompt: PromptMessages, query: str, chunks_used: int) -> Answer:
    """
    Call the OpenAI API with the prompt messages.
    Parse citation strings from the response (lines matching '[Source: ..., Section: ...]').
    Return an Answer dataclass.
    Model: gpt-4o-mini
    Max tokens: 1024
    Read OPENAI_API_KEY from environment variables using python-dotenv.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ],
    )

    text = response.choices[0].message.content or ""
    citations = re.findall(r'\[Source: .+?, Section: .+?\]', text)

    return Answer(
        text=text,
        citations=citations,
        query=query,
        chunks_used=chunks_used,
    )

def generate(prompt: PromptMessages, query: str, chunks_used: int) -> Answer:
    """
    Call the Anthropic API with the prompt messages.
    Parse citation strings from the response (lines matching '[Source: ..., Section: ...]').
    Return an Answer dataclass.
    Model: claude-haiku-4-5
    Max tokens: 1024
    Read ANTHROPIC_API_KEY from environment variables using python-dotenv.
    """
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=prompt.system,
        messages=[
            {"role": "user", "content": prompt.user},
        ],
    )

    text = "".join(block.text for block in response.content if block.type == "text")
    citations = re.findall(r'\[Source: .+?, Section: .+?\]', text)

    return Answer(
        text=text,
        citations=citations,
        query=query,
        chunks_used=chunks_used,
    )


def generate_with_litellm(prompt: PromptMessages, query: str, chunks_used: int, model: str | None = None) -> Answer:
    """
    Call the LiteLLM gateway's OpenAI-compatible chat endpoint.
    model is a gateway alias (e.g. gemini-2.5-flash); defaults to settings.DEFAULT_CHAT_MODEL.
    """
    client = OpenAI(base_url=settings.LITELLM_BASE_URL, api_key=settings.LITELLM_API_KEY or "missing")

    response = client.chat.completions.create(
        model=model or settings.DEFAULT_CHAT_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ],
    )

    text = response.choices[0].message.content or ""
    return Answer(
        text=text,
        citations=re.findall(CITATION_PATTERN, text),
        query=query,
        chunks_used=chunks_used,
    )


def generate_answer(prompt: PromptMessages, query: str, chunks_used: int, model: str | None = None) -> Answer:
    """Dispatch to the generator selected by settings.GENERATOR (litellm | anthropic | openai)."""
    if settings.GENERATOR == "litellm":
        return generate_with_litellm(prompt, query, chunks_used, model)
    if settings.GENERATOR == "anthropic":
        return generate(prompt, query, chunks_used)
    if settings.GENERATOR == "openai":
        return generateUsingOpenAI(prompt, query, chunks_used)
    raise ValueError(f"Unknown RAG_GENERATOR '{settings.GENERATOR}'")
