from .references import (
    FORMAT_REFERENCES,
    EmptyReferenceStyle,
    HTMLReferenceStyle,
    MarkdownReferenceStyle,
    ReferenceStyle,
    TextReferenceStyle,
    manage_references,
)

__all__ = [
    "ReferenceStyle",
    "EmptyReferenceStyle",
    "TextReferenceStyle",
    "HTMLReferenceStyle",
    "MarkdownReferenceStyle",
    "manage_references",
    "FORMAT_REFERENCES",
]

from importlib import metadata

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
