from __future__ import annotations


class IngestionFailure(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class RetryableIngestionFailure(IngestionFailure):
    pass


class LeaseLost(RuntimeError):
    """The worker no longer owns the job lease and must stop writing."""


class CorpusCapacityExceeded(IngestionFailure):
    def __init__(self) -> None:
        super().__init__("CORPUS_LIMIT_EXCEEDED", "The indexed chunk capacity has been reached.")
