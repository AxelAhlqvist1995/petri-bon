"""Load prompt templates bundled as package data.

Templates are addressed as ``category/name`` relative to this package, e.g.
``load_prompt("preference/transcript", "basic_no_prefill")``. Callers may also
pass a filesystem path or inline template text instead of a bundled name.
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from pathlib import Path


@cache
def load_prompt(category: str, name: str) -> str:
    resource = files("petri_bon.prompts").joinpath(category, f"{name}.txt")
    return resource.read_text(encoding="utf-8").rstrip("\n")


def resolve_prompt(category: str, spec: str) -> str:
    """Resolve a prompt spec: bundled name, filesystem path, or inline text.

    Anything containing a newline or ``{`` is treated as inline template text;
    an existing file path is read; otherwise the spec is a bundled name.
    """
    if "\n" in spec or "{" in spec:
        return spec
    path = Path(spec)
    if path.suffix == ".txt" and path.exists():
        return path.read_text(encoding="utf-8").rstrip("\n")
    return load_prompt(category, spec)
