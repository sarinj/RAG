class ServiceError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class DatabaseUnavailable(RuntimeError):
    pass


class DatabaseSchemaUnavailable(RuntimeError):
    pass


class ProviderError(RuntimeError):
    pass


class ProviderTimeout(ProviderError):
    pass


class ProviderModelMissing(ProviderError):
    pass


class ProviderProtocolError(ProviderError):
    pass
