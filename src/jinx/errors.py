"""J errors for incorrect usage of J primitives."""


class BaseJError(Exception):
    pass


class LengthError(BaseJError):
    pass


class DomainError(BaseJError):
    pass


class ValenceError(BaseJError):
    pass


class JIndexError(BaseJError):
    pass


class SpellingError(BaseJError):
    pass


class StackError(BaseJError):
    pass


class JinxNotImplementedError(BaseJError):
    """Raised when a feature is not implemented in Jinx."""
