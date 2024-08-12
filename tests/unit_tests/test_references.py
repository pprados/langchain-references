from typing import Generator, List, Tuple, cast

from langchain_core.documents import Document
from langchain_core.documents.base import BaseMedia
from langchain_core.messages import AIMessageChunk, BaseMessage

from langchain_references import (
    EmptyReferenceStyle,
    HTMLReferenceStyle,
    MarkdownReferenceStyle,
    TextReferenceStyle,
)
from langchain_references.references import ReferenceStyle, _manage_references

_four_documents = [
    Document(
        page_content="doc1",
        id="1",
        metadata={"source": "a.html#chap1", "row": 1, "title": "doc1"},
    ),
    Document(
        page_content="doc2",
        id="2",
        metadata={"source": "a.html#chap2", "row": 2, "title": "doc2"},
    ),
    Document(
        page_content="doc3",
        id="3",
        metadata={"source": "b.pdf", "row": 3, "title": "doc3"},
    ),
    Document(
        page_content="doc4",
        id="4",
        metadata={"source": "b.pdf", "row": 4, "title": "doc4"},
    ),
    Document(
        page_content="doc5",
        id="5",
        metadata={"source": "c.csv", "row": 5, "title": "doc5"},
    ),
]

_two_documents = _four_documents[:2]


class TestReferenceStyle(MarkdownReferenceStyle):
    def format_reference(self, ref: int, media: BaseMedia) -> str:
        return f"[{ref}]({media.metadata['source']})"

    def format_all_references(self, refs: List[Tuple[int, BaseMedia]]) -> str:
        result = ["\n"]
        for ref, media in refs:
            result.append(
                f"- {ref} "
                f"[{media.metadata['title']}]"
                f"({media.metadata['source']}"
                f"#{media.metadata['row']})\n"
            )
        return "".join(result)


def _send(
    references: Generator[BaseMessage | None, AIMessageChunk | None, None],
    content: str | None,
) -> str | None:
    result: BaseMessage | None
    if content is not None:
        result = references.send(AIMessageChunk(content=content))
    else:
        result = references.send(None)  # FIXME: None ou '' ? Simplifie le typage
    if result:
        return cast(str | None, result.content)
    return cast(str | None, result)


def test_single_token() -> None:
    manage_references = _manage_references(
        style=TestReferenceStyle(), medium=_two_documents
    )

    _send(manage_references, None)  # Start generator
    assert _send(manage_references, "Hello [1](id=1) world  [2](id=2)") == "Hello "
    assert _send(manage_references, "") == "[1](a.html#chap1) world[2](a.html#chap2)"
    assert (
        _send(manage_references, None) == "\n"
        "- 1 [doc1](a.html#chap1#1)\n"
        "- 2 [doc2](a.html#chap2#2)\n"
    )


def test_split_token() -> None:
    manage_references = _manage_references(
        style=TestReferenceStyle(), medium=_two_documents
    )

    _send(manage_references, None)
    assert _send(manage_references, "Hello [") == "Hello "
    assert _send(manage_references, "1](id=") is None
    assert _send(manage_references, "1)") == "[1](a.html#chap1)"
    assert _send(manage_references, None) == "\n- 1 [doc1](a.html#chap1#1)\n"


def test_windows_large() -> None:
    manage_references = _manage_references(
        style=TestReferenceStyle(), medium=_two_documents
    )

    # Test if windows_str is bigger than _MAX_WINDOWS_SIZE
    _send(manage_references, None)
    assert _send(manage_references, "Hello [") == "Hello "
    assert _send(manage_references, "01234567890123456789") == "[01234567890123456789"
    assert _send(manage_references, None) == "\n"


def test_windows_not_empty_at_end() -> None:
    manage_references = _manage_references(
        style=TestReferenceStyle(), medium=_two_documents
    )

    # Test if windows_str not empty at the end
    _send(manage_references, None)
    assert _send(manage_references, "Hello [") == "Hello "
    assert (
        _send(manage_references, "[1](id=1)[2](id=2)7890")
        == "[[1](a.html#chap1)[2](a.html#chap2)"
    )
    assert (
        _send(manage_references, None) == "7890\n"
        "- 1 [doc1](a.html#chap1#1)\n"
        "- 2 [doc2](a.html#chap2#2)\n"
    )


def test_manage_complex_scenario() -> None:
    manage_references = _manage_references(
        style=TestReferenceStyle(), medium=_four_documents
    )

    _send(manage_references, None)
    assert (
        _send(
            manage_references,
            "yes[1](id=3), maybe[2](id=2), no[3](id=4), yes[4](id=1), error[5](id=10)",
        )
        == "yes"
    )
    assert (
        _send(manage_references, "") == "[1](b.pdf), "
        "maybe[2](a.html#chap2), "
        "no[1](b.pdf), "
        "yes[3](a.html#chap1), "
        "error"
    )
    assert (
        _send(manage_references, None) == "\n"
        "- 1 [doc4](b.pdf#4)\n"
        "- 2 [doc2](a.html#chap2#2)\n"
        "- 3 [doc1](a.html#chap1#1)\n"
    )


def test_NUMBER() -> None:
    manage_references = _manage_references(
        style=TestReferenceStyle(), medium=_four_documents
    )

    _send(manage_references, None)
    _send(manage_references, "[NUMBER](id=1)")
    assert _send(manage_references, "") == "[1](a.html#chap1)"


def test_style_empty() -> None:
    documents: List[BaseMedia] = [
        Document(
            page_content="doc1",
            id="1",
            metadata={"source": "source1", "row": 1, "title": "title1"},
        ),
        Document(
            page_content="doc2",
            id="2",
            metadata={"source": "source2", "row": 2, "title": "title2"},
        ),
    ]
    manage_references = _manage_references(
        style=EmptyReferenceStyle(), medium=documents
    )

    # Test with title
    _send(manage_references, None)
    assert (
        _send(
            manage_references,
            "yes[1](id=3), maybe[2](id=2), no[3](id=4), yes[4](id=1), error[5](id=10)",
        )
        == "yes"
    )
    assert _send(manage_references, "") == ", maybe, no, yes, error"
    assert _send(manage_references, None) == ""

    # Test without title
    documents[0].metadata.pop("title")
    documents[1].metadata.pop("title")
    _send(manage_references, None)
    assert (
        _send(
            manage_references,
            "yes[1](id=3), maybe[2](id=2), "
            "no[3](id=4), yes[4](id=1), error[5](id=10)",
        )
        == "yes"
    )
    assert _send(manage_references, "") == ", maybe, no, yes, error"
    assert _send(manage_references, None) == ""


def test_style_text() -> None:
    documents: List[BaseMedia] = [
        Document(
            page_content="doc1",
            id="1",
            metadata={"source": "source1", "row": 1, "title": "title1"},
        ),
        Document(
            page_content="doc2",
            id="2",
            metadata={"source": "source2", "row": 2, "title": "title2"},
        ),
    ]
    manage_references = _manage_references(style=TextReferenceStyle(), medium=documents)

    # Test with title
    _send(manage_references, None)
    assert (
        _send(
            manage_references,
            "yes[1](id=3), maybe[2](id=2), "
            "no[3](id=4), yes[4](id=1), error[5](id=10)",
        )
        == "yes"
    )
    assert _send(manage_references, "") == ", maybe[1], no, yes[2], error"
    assert (
        _send(manage_references, None) == "\n\n"
        "- [1] title2 (source2)\n"
        "- [2] title1 (source1)\n"
    )

    # Test without title
    documents[0].metadata.pop("title")
    documents[1].metadata.pop("title")
    manage_references.send(None)
    assert (
        _send(
            manage_references,
            "yes[1](id=3), maybe[2](id=2), "
            "no[3](id=4), yes[4](id=1), error[5](id=10)",
        )
        == "yes"
    )
    assert _send(manage_references, "") == ", maybe[1], no, yes[2], error"
    assert _send(manage_references, None) == "\n\n" "- [1] source2\n" "- [2] source1\n"


def test_style_markdown() -> None:
    documents: List[BaseMedia] = [
        Document(
            page_content="doc1",
            id="1",
            metadata={"source": "source1", "row": 1, "title": "title1"},
        ),
        Document(
            page_content="doc2",
            id="2",
            metadata={"source": "source2", "row": 2, "title": "title2"},
        ),
    ]
    manage_references = _manage_references(
        style=MarkdownReferenceStyle(), medium=documents
    )

    # Test with title
    _send(manage_references, None)
    assert (
        _send(
            manage_references,
            "yes[1](id=3), maybe[2](id=2), "
            "no[3](id=4), yes[4](id=1), error[5](id=10)",
        )
        == "yes"
    )
    assert (
        _send(manage_references, "")
        == ", maybe<sup>[[1](source2)]</sup>, no, yes<sup>[[2](source1)]</sup>, error"
    )
    assert (
        _send(manage_references, None) == "\n\n"
        "- **1** [title2](source2)\n"
        "- **2** [title1](source1)\n"
    )

    # Test without title
    documents[0].metadata.pop("title")
    documents[1].metadata.pop("title")
    _send(manage_references, None)
    assert (
        _send(
            manage_references,
            "yes[1](id=3), maybe[2](id=2), "
            "no[3](id=4), yes[4](id=1), error[5](id=10)",
        )
        == "yes"
    )
    assert (
        _send(manage_references, "")
        == ", maybe<sup>[[1](source2)]</sup>, no, yes<sup>[[2](source1)]</sup>, error"
    )
    assert (
        _send(manage_references, None) == "\n\n"
        "- **1** <source2>\n"
        "- **2** <source1>\n"
    )


def test_style_html() -> None:
    documents: List[BaseMedia] = [
        Document(
            page_content="doc1",
            id="1",
            metadata={"source": "source1", "row": 1, "title": "title1"},
        ),
        Document(
            page_content="doc2",
            id="2",
            metadata={"source": "source2", "row": 2, "title": "title2"},
        ),
    ]

    # Test with title
    manage_references = _manage_references(style=HTMLReferenceStyle(), medium=documents)
    _send(manage_references, None)
    assert (
        _send(
            manage_references,
            "yes[1](id=3), maybe[2](id=2), "
            "no[3](id=4), yes[4](id=1), error[5](id=10)",
        )
        == "yes"
    )
    assert (
        _send(manage_references, "")
        == ', maybe<sup><a href="source2">1</a></sup>, no, yes<sup><a '
        'href="source1">2</a></sup>, error'
    )
    assert (
        _send(manage_references, None) == '\n<ol><li><a href="source2">title2</a></li>'
        '<li><a href="source1">title1</a></li></ol>'
    )

    # Test without title
    documents[0].metadata.pop("title")
    documents[1].metadata.pop("title")
    _send(manage_references, None)
    assert (
        _send(
            manage_references,
            "yes[1](id=3), maybe[2](id=2), "
            "no[3](id=4), yes[4](id=1), error[5](id=10)",
        )
        == "yes"
    )
    assert (
        _send(manage_references, "")
        == ', maybe<sup><a href="source2">1</a></sup>, no, yes<sup><a '
        'href="source1">2</a></sup>, error'
    )
    assert (
        _send(manage_references, None) == '\n<ol><li><a href="source2">source2</a></li>'
        '<li><a href="source1">source1</a></li></ol>'
    )


def test_my_style() -> None:
    def my_source(media: BaseMedia) -> str:
        return f'{media.metadata["source"]}#{media.metadata["row"]}'

    class MyReferenceStyle(ReferenceStyle):
        source_id_key = my_source

        def format_reference(self, ref: int, media: BaseMedia) -> str:
            return f"[{media.metadata['title']}]"

        def format_all_references(self, refs: List[Tuple[int, BaseMedia]]) -> str:
            if not refs:
                return ""
            result = []
            for ref, media in refs:
                source = self.source_id_key.__func__(media)  # type: ignore
                result.append(f"- [{ref}] {source}\n")
            if not result:
                return ""
            return "\n\n" + "".join(result)

    documents: List[BaseMedia] = [
        Document(
            page_content="doc1",
            id="1",
            metadata={"source": "source1", "row": 1, "title": "title1"},
        ),
        Document(
            page_content="doc2",
            id="2",
            metadata={"source": "source2", "row": 2, "title": "title2"},
        ),
    ]

    # Test with title
    manage_references = _manage_references(style=MyReferenceStyle(), medium=documents)
    _send(manage_references, None)
    assert (
        _send(
            manage_references,
            "yes[1](id=3), maybe[2](id=2), "
            "no[3](id=4), yes[4](id=1), error[5](id=10)",
        )
        == "yes"
    )
    assert _send(manage_references, "") == ", maybe[title2], no, yes[title1], error"
    assert (
        _send(manage_references, None) == "\n\n" "- [1] source2#2\n" "- [2] source1#1\n"
    )
