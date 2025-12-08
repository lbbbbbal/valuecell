"""Decision making components."""

__all__ = ["BaseComposer", "LlmComposer"]


def __getattr__(name):
    if name == "BaseComposer":
        from .interfaces import BaseComposer

        return BaseComposer

    if name == "LlmComposer":
        from .prompt_based.composer import LlmComposer

        return LlmComposer

    raise AttributeError(name)
