"""J errors for incorrect usage of J primitives."""


class BaseJError(Exception):
    pass


class LengthError(BaseJError):
    pass


class DomainError(BaseJError):
    pass


class ValenceError(BaseJError):
    pass
