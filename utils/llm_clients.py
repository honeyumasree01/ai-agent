from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from utils.settings import Settings


def claude(settings: Settings) -> BaseChatModel:
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=settings.anthropic_api_key,
    )


def gpt4o(settings: Settings) -> BaseChatModel:
    return ChatOpenAI(
        model="gpt-4o-2024-05-13",
        api_key=settings.openai_api_key,
    )


def gemini(settings: Settings) -> BaseChatModel:
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-001",
        google_api_key=settings.google_api_key,
    )
