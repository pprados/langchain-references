from __future__ import annotations

from typing import Any, Callable, Generator, Iterator, cast
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.documents.base import BaseMedia
from langchain_core.language_models import LanguageModelInput, LanguageModelOutput
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig

from langchain_references import (
    EmptyReferenceStyle,
    HTMLReferenceStyle,
    MarkdownReferenceStyle,
    TextReferenceStyle,
    manage_references,
)
from langchain_references.references import (
    _PREFIX as _P,
)
from langchain_references.references import (
    _SUFFIX as _S,
)
from langchain_references.references import (
    ReferenceStyle,
    _manage_references,
)


class _TestRunnable(Runnable[LanguageModelInput, LanguageModelOutput]):
    text_fragments: list[str]

    def __init__(self, text_fragments: list[str]) -> None:
        self.text_fragments = text_fragments

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> LanguageModelOutput:
        raise NotImplementedError()

    def stream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[LanguageModelOutput]:
        for text_fragment in self.text_fragments:
            yield text_fragment


def collect_fragments(
    text_fragments: list[str],
    documents: list[Document],
    style: ReferenceStyle = MarkdownReferenceStyle(),
) -> str:
    return "".join(
        [
            r.content  # type: ignore
            for r in manage_references(
                _TestRunnable(text_fragments=text_fragments), style=style
            ).stream(
                {"documents": documents}  # type: ignore
            )
        ]
    )


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

    def format_all_references(self, refs: list[tuple[int, BaseMedia]]) -> str:
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
        result = references.send(None)
    if result:
        return cast(str | None, result.content)
    return cast(str | None, result)


def test_single_token() -> None:
    manage_references = _manage_references(
        style=TestReferenceStyle(), medium=_two_documents
    )

    _send(manage_references, None)  # Start generator
    assert _send(manage_references, f"Hello {_P}1{_S} world  " f"{_P}2{_S}") == "Hello "
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
    assert _send(manage_references, f"Hello {_P}") == "Hello "
    assert _send(manage_references, "1") is None
    assert _send(manage_references, f"{_S}") == "[1](a.html#chap1)"
    assert _send(manage_references, None) == "\n- 1 [doc1](a.html#chap1#1)\n"


def test_windows_large() -> None:
    assert (
        collect_fragments(
            [
                f"Hello {_P}",
                "01234567890123456789",
                "\n",
            ],
            _two_documents,
            TestReferenceStyle(),
        )
        == f"Hello {_P}01234567890123456789\n"
    )

    assert (
        collect_fragments(
            [
                f"Hello {_P}",
                f"0123456789012345678{_P}",
                f"1{_S}\n",
            ],
            _two_documents,
            TestReferenceStyle(),
        )
        == f"Hello {_P}0123456789012345678[1](a.html#chap1)\n\n"
        "- 1 [doc1](a.html#chap1#1)\n"
    )

    assert (
        collect_fragments(
            [
                f"Hello {_P}",
                f"01234567890123456{_P}1{_S}",
            ],
            _two_documents,
            TestReferenceStyle(),
        )
        == f"Hello {_P}01234567890123456[1](a.html#chap1)\n"
        "- 1 [doc1](a.html#chap1#1)\n"
    )

    assert (
        collect_fragments(
            [
                f"Hello {_P}",
                f"01234567890123456{_P}1{_S}",
            ],
            _two_documents,
            TestReferenceStyle(),
        )
        == f"Hello {_P}01234567890123456[1](a.html#chap1)\n"
        "- 1 [doc1](a.html#chap1#1)\n"
    )


def test_windows_not_empty_at_end() -> None:
    manage_references = _manage_references(
        style=TestReferenceStyle(), medium=_two_documents
    )

    # Test if windows_str not empty at the end
    _send(manage_references, None)
    assert _send(manage_references, f"Hello {_P}") == "Hello "
    assert (
        _send(manage_references, f"{_P}1{_S}" f"{_P}2{_S}" f"1234567890")
        == f"{_P}[1](a.html#chap1)[2](a.html#chap2)"
    )
    assert (
        _send(manage_references, None) == "1234567890\n"
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
            f"yes{_P}1{_S}, maybe{_P}2{_S}, no{_P}3{_S}, "
            f"yes{_P}4{_S}, error{_P}5{_S}",
        )
        == "yes"
    )
    assert (
        _send(manage_references, "") == "[1](a.html#chap1), "
        "maybe[2](a.html#chap2), "
        "no[3](b.pdf), "
        "yes[3](b.pdf), "
        "error[4](c.csv)"
    )
    assert (
        _send(manage_references, None) == "\n"
        "- 1 [doc1](a.html#chap1#1)\n"
        "- 2 [doc2](a.html#chap2#2)\n"
        "- 3 [doc4](b.pdf#4)\n"
        "- 4 [doc5](c.csv#5)\n"
    )


def test_manage_fake_pattern() -> None:
    assert (
        collect_fragments(
            [
                f"read page « {_P}foo{_S}(https://www.foo.org) »",
            ],
            _two_documents,
        )
        == f"read page « {_P}foo{_S}(https://www.foo.org) »"
    )


@patch("langchain_references.references.logger")
def test_manage_invalid_reference(mock_logging: Any) -> None:
    assert (
        collect_fragments(
            [
                f"before {_P}99{_S} after",
            ],
            [],
        )
        == "before  after"
    )
    assert mock_logging.warning.call_count == 1


def test_NUMBER() -> None:
    manage_references = _manage_references(
        style=TestReferenceStyle(), medium=_four_documents
    )

    _send(manage_references, None)
    _send(manage_references, f"{_P}NUMBER{_S}") == ""
    assert _send(manage_references, "") == ""


def test_style_empty() -> None:
    documents: list[BaseMedia] = [
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
            f"yes{_P}3{_S}, maybe{_P}2{_S}, no{_P}4{_S}, "
            f"yes{_P}1{_S}, error{_P}10{_S}",
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
            f"yes{_P}3{_S}, maybe{_P}2{_S}, "
            f"no{_P}4{_S}, yes{_P}1{_S}, error{_P}10{_S}",
        )
        == "yes"
    )
    assert _send(manage_references, "") == ", maybe, no, yes, error"
    assert _send(manage_references, None) == ""


def test_style_text() -> None:
    documents: list[BaseMedia] = [
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
            f"yes{_P}3{_S}, maybe{_P}2{_S}, "
            f"no{_P}4{_S}, yes{_P}1{_S}, error{_P}10{_S}",
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
            f"yes{_P}3{_S}, maybe{_P}2{_S}, "
            f"no{_P}4{_S}, yes{_P}1{_S}, error{_P}10{_S}",
        )
        == "yes"
    )
    assert _send(manage_references, "") == ", maybe[1], no, yes[2], error"
    assert _send(manage_references, None) == "\n\n" "- [1] source2\n" "- [2] source1\n"


def test_style_markdown_compatible() -> None:
    documents: list[BaseMedia] = [
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
        style=MarkdownReferenceStyle(foot_note_compatibe=True), medium=documents
    )

    # Test with title
    _send(manage_references, None)
    assert (
        _send(
            manage_references,
            f"yes{_P}3{_S}, maybe{_P}2{_S}, "
            f"no{_P}4{_S}, yes{_P}1{_S}, error{_P}10{_S}",
        )
        == "yes"
    )
    assert _send(manage_references, "") == ", maybe[^1], no, yes[^2], error"
    assert (
        _send(manage_references, None) == "\n\n"
        "[^1]: [title2](source2)\n"
        "[^2]: [title1](source1)\n"
    )

    # Test without title
    documents[0].metadata.pop("title")
    documents[1].metadata.pop("title")
    _send(manage_references, None)
    assert (
        _send(
            manage_references,
            f"yes{_P}3{_S}, maybe{_P}2{_S}, "
            f"no{_P}4{_S}, yes{_P}1{_S}, error{_P}10{_S}",
        )
        == "yes"
    )
    assert _send(manage_references, "") == ", maybe[^1], no, yes[^2], error"
    assert _send(manage_references, None) == "\n\n" "[^1]: source2\n" "[^2]: source1\n"


def test_style_markdown_not_compatible() -> None:
    documents: list[BaseMedia] = [
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
        style=MarkdownReferenceStyle(foot_note_compatibe=False), medium=documents
    )

    # Test with title
    _send(manage_references, None)
    assert (
        _send(
            manage_references,
            f"yes{_P}3{_S}, maybe{_P}2{_S}, "
            f"no{_P}4{_S}, yes{_P}1{_S}, error{_P}10{_S}",
        )
        == "yes"
    )
    assert (
        _send(manage_references, "") == ", "
        'maybe<a href="#fn1" id="1">[1]</a></sup>, '
        'no, yes<a href="#fn2" id="2">[2]</a></sup>, '
        "error"
    )
    assert (
        _send(manage_references, None) == "\n\n"
        '<sup id="fn1" style="font-size: 0.7em;">1.</a> '
        "[title2](source2)</sup></small>  \n"
        '<sup id="fn2" style="font-size: 0.7em;">2.</a> '
        "[title1](source1)</sup></small>  \n"
    )

    # Test without title
    documents[0].metadata.pop("title")
    documents[1].metadata.pop("title")
    _send(manage_references, None)
    assert (
        _send(
            manage_references,
            f"yes{_P}3{_S}, maybe{_P}2{_S}, "
            f"no{_P}4{_S}, yes{_P}1{_S}, error{_P}10{_S}",
        )
        == "yes"
    )
    assert (
        _send(manage_references, "") == ', maybe<a href="#fn1" id="1">[1]</a></sup>, '
        "no, "
        'yes<a href="#fn2" id="2">[2]</a></sup>, '
        "error"
    )
    assert (
        _send(manage_references, None) == "\n\n"
        '<sup id="fn1" style="font-size: 0.7em;">1.</a> source2</sup>  \n'
        '<sup id="fn2" style="font-size: 0.7em;">2.</a> source1</sup>  \n'
    )


def test_style_html() -> None:
    documents: list[BaseMedia] = [
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
            f"yes{_P}3{_S}, maybe{_P}2{_S}, "
            f"no{_P}4{_S}, yes{_P}1{_S}, error{_P}10{_S}",
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
            f"yes{_P}3{_S}, maybe{_P}2{_S}, "
            f"no{_P}4{_S}, yes{_P}1{_S}, error{_P}10{_S}",
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
    # Test My style, with the exclusion of big documents (total_pages > 5)
    def my_source(media: BaseMedia) -> str:
        if "row" in media.metadata:
            return f'{media.metadata["source"]}#{media.metadata["row"]}'
        return media.metadata["source"]

    class MyReferenceStyle(ReferenceStyle):
        source_id_key: Callable[[BaseMedia], str] = my_source

        def format_reference(self, ref: int, media: BaseMedia) -> str | None:
            get_total_pages = self._get_key_assigner(self.total_pages_key)
            total_pages = get_total_pages(media)
            if total_pages and total_pages > self.max_total_pages:
                return None
            return f"[{media.metadata['title']}]"

        def format_all_references(self, refs: list[tuple[int, BaseMedia]]) -> str:
            if not refs:
                return ""
            result = []
            for ref, media in refs:
                source = self.source_id_key.__func__(media)  # type: ignore
                result.append(f"- [{ref}] {source}\n")
            if not result:
                return ""
            return "\n\n" + "".join(result)

    documents: list[BaseMedia] = [
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
        Document(
            page_content="doc3",
            id="3",
            metadata={"source": "source3", "total_pages": 200, "title": "title3"},
        ),
    ]

    # Test with title
    manage_references = _manage_references(style=MyReferenceStyle(), medium=documents)
    _send(manage_references, None)
    assert (
        _send(
            manage_references,
            f"yes{_P}3{_S}, maybe{_P}2{_S}, "
            f"no{_P}4{_S}, yes{_P}1{_S}, remove{_P}3{_S}, "
            f"error{_P}10{_S}",
        )
        == "yes"
    )
    assert _send(manage_references, "") == (
        ", maybe[title2], no, yes[title1], remove, error"
    )
    assert _send(manage_references, None) == "\n\n- [1] source2#2\n- [2] source1#1\n"
