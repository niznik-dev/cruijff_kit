"""Template registry and factory."""

from .base import BaseTemplate
from .dictionary import DictionaryTemplate
from .narrative import NarrativeTemplate

__all__ = ["BaseTemplate", "get_template"]


def get_template(
    template_type: str,
    template_file: str | None = None,
    cache_path: str | None = None,
    style_guidance: str | None = None,
) -> BaseTemplate:
    """Create a template instance by type name.

    Args:
        template_type: "dictionary", "narrative", or "llm_narrative"
        template_file: Path to a Jinja2 template file (narrative only)
        cache_path: Path to API response cache (llm_narrative only)
        style_guidance: Style instructions for llm_narrative (llm_narrative only)

    Returns:
        A BaseTemplate instance.
    """
    if template_type == "dictionary":
        return DictionaryTemplate()
    elif template_type == "narrative":
        return NarrativeTemplate(template_file=template_file)
    elif template_type == "llm_narrative":
        from .llm_narrative import LLMNarrativeTemplate

        return LLMNarrativeTemplate(
            cache_path=cache_path, style_guidance=style_guidance
        )
    else:
        raise ValueError(
            f"Unknown template type '{template_type}'. "
            f"Choose from: dictionary, narrative, llm_narrative"
        )
