"""Shared error types for LLM cascade and agent flows."""


class AllModelsFailedError(Exception):
    """Raised when every model in the cascade fails after retries."""

    def __init__(self, message: str = "All LLM providers failed") -> None:
        super().__init__(message)
