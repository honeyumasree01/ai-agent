class AllModelsFailedError(Exception):
    def __init__(self, message: str = "All LLM providers failed") -> None:
        super().__init__(message)
