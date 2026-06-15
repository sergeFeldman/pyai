"""Explainable marker and mixin for domain models participating in the explanation workflow."""

from typing import Annotated, get_type_hints


class Explainable:
    """Marker for dataclass fields that support attribute-level explanation.

    Apply via Annotated to any field whose value the explanation engine
    should be able to interpret and explain to the user:

        status: Annotated[ClaimStatus, Explainable()]

    Fields without this marker are treated as non-explainable and will
    be rejected if included in an explanation request.
    """


class ExplainableMixin:
    """Mixin that exposes explainable_attributes() for any dataclass.

    Derives the set of explainable field names at runtime by inspecting
    Annotated metadata. Any field marked with Explainable() is included.

    Usage:

        @dataclass
        class Claim(ExplainableMixin):
            status: Annotated[ClaimStatus, Explainable()]
            is_fraud: Annotated[bool, Explainable()] = False

        Claim.explainable_attributes()  # {"status", "is_fraud"}

    Inherit this mixin in any domain model that participates in the
    explanation workflow.
    """

    @classmethod
    def explainable_attributes(cls) -> set[str]:
        """Return the set of field names marked as Explainable.

        Returns:
            set[str]: Field names eligible for explanation.
        """
        return {
            name for name, hint in get_type_hints(cls, include_extras=True).items()
            if any(isinstance(m, Explainable) for m in getattr(hint, "__metadata__", []))
        }
