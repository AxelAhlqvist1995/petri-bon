"""Registry imports for inspect_ai entry-point discovery.

Referenced by the ``inspect_ai`` entry point in pyproject.toml so
``inspect eval petri_bon/bon_audit`` resolves without an explicit import.
"""

from .tasks import bon_audit  # noqa: F401
