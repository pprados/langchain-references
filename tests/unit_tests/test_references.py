from __future__ import annotations

from typing import Any, Generator, Iterator, List, Optional, Tuple, cast
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
    ReferenceStyle,
    _manage_references,
)
from langchain_references.references import (
    _PREFIX as _P,
)
from langchain_references.references import (
    _SUFFIX as _S,
)


class _TestRunnable(Runnable[LanguageModelInput, LanguageModelOutput]):
    text_fragments: List[str]

    def __init__(self, text_fragments: List[str]) -> None:
        self.text_fragments = text_fragments

    def invoke(
            self,
            input: LanguageModelInput,
            config: Optional[RunnableConfig] = None,
            **kwargs: Any,
    ) -> LanguageModelOutput:
        raise NotImplementedError()

    def stream(
            self,
            input: LanguageModelInput,
            config: Optional[RunnableConfig] = None,
            **kwargs: Optional[Any],
    ) -> Iterator[LanguageModelOutput]:
        for text_fragment in self.text_fragments:
            yield text_fragment


def collect_fragments(
        text_fragments: List[str],
        documents: List[Document],
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
        return cast(Optional[str], result.content)
    return cast(Optional[str], result)


def test_single_token() -> None:
    manage_references = _manage_references(
        style=TestReferenceStyle(), medium=_two_documents
    )

    _send(manage_references, None)  # Start generator
    assert (
            _send(manage_references,
                  f"Hello {_P}1{_S}(id=1) world  " f"{_P}2{_S}(id=2)")
            == "Hello "
    )
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
    assert _send(manage_references, f"1{_S}(id=") is None
    assert _send(manage_references, "1)") == "[1](a.html#chap1)"
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
                    f"1{_S}(id=1)\n",
                ],
                _two_documents,
                TestReferenceStyle(),
            )
            == f"Hello {_P}0123456789012345678[1](a.html#chap1)\n"
               "\n"
               "- 1 [doc1](a.html#chap1#1)\n"
    )

    assert (
            collect_fragments(
                [
                    f"Hello {_P}",
                    f"01234567890123456{_P}1{_S}",
                    "(id=1)\n",
                ],
                _two_documents,
                TestReferenceStyle(),
            )
            == f"Hello {_P}01234567890123456[1](a.html#chap1)\n"
               "\n"
               "- 1 [doc1](a.html#chap1#1)\n"
    )

    assert (
            collect_fragments(
                [
                    f"Hello {_P}",
                    f"01234567890123456{_P}1{_S}",
                    "(id=1)\n",
                ],
                _two_documents,
                TestReferenceStyle(),
            )
            == f"Hello {_P}01234567890123456[1](a.html#chap1)\n"
               "\n"
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
            _send(manage_references, f"{_P}1{_S}(id=1)"
                                     f"{_P}2{_S}(id=2)"
                                     f"1234567890")
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
                f"yes{_P}1{_S}(id=3), maybe{_P}2{_S}(id=2), no{_P}3{_S}(id=4), "
                f"yes{_P}4{_S}(id=1), error{_P}5{_S}(id=10)",
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


# def test_TOTO() -> None:
#     chunks = ['', 'Sel', 'on', ' le', ' document', ' "', 'Mes', ' interruptions', ' de',
#               ' car', 'rière', '"', ' [', '<[', '1', ']>', '](', 'id', '=', '1', '),',
#               ' le', ' RSA', ' (', 'Re', 'ven', 'u', ' de', ' Solid', 'ar', 'ité',
#               ' Active', ')', ' ne', ' cr', 'ée', ' pas', ' de', ' droit', ' à', ' la',
#               ' re', 'tra', 'ite', ' et', ' ne', ' permet', ' pas', ' de', ' val',
#               'ider', ' des', ' trimest', 'res', '.', ' C', 'ela', ' sign', 'ifie',
#               ' que', ' les', ' péri', 'odes', ' de', ' RSA', ' ne', ' sont', ' pas',
#               ' pr', 'ises', ' en', ' compte', ' dans', ' le', ' calcul', ' de', ' la',
#               ' re', 'tra', 'ite', ' en', ' term', 'es', ' de', ' trimest', 'res',
#               ' valid', 'és', '.\n\n', 'C', 'epend', 'ant', ',', ' il', ' est', ' préc',
#               'isé', ' que', ' si', ' vous', ' perce', 'vez', ' le', ' RSA', ',',
#               ' votre', ' ca', 'isse', ' rég', 'ionale', ' vous', ' contact', 'e',
#               ' lorsque', ' vous', ' avez', ' droit', ' à', ' votre', ' re', 'tra',
#               'ite', ' person', 'nelle', ',', ' à', ' une', ' re', 'tra', 'ite', ' de',
#               ' ré', 'version', ' ou', ' à', ' une', ' allocation', ' de', ' ve', 'uv',
#               'age', ' [', '<[', '2', ']>', '](', 'id', '=', '1', ').\n\n', 'Il',
#               ' est', ' donc', ' important', ' de', ' not', 'er', ' que', ' les',
#               ' péri', 'odes', ' de', ' RSA', ' ne', ' sont', ' pas', ' direct',
#               'ement', ' pr', 'ises', ' en', ' compte', ' dans', ' le', ' calcul',
#               ' de', ' la', ' re', 'tra', 'ite', ',', ' mais', ' il', ' est',
#               ' possible', ' que', ' votre', ' ca', 'isse', ' rég', 'ionale', ' vous',
#               ' contact', 'e', ' pour', ' vous', ' inform', 'er', ' de', ' votre',
#               ' droit', ' à', ' la', ' re', 'tra', 'ite', '.', '']
#     conv_chunk = [
#         r.content  # type: ignore
#         for r in manage_references(
#             _TestRunnable(text_fragments=chunks), style=MarkdownReferenceStyle()
#         ).stream(
#             {"documents": _two_documents}  # type: ignore
#         )
#     ]
#
#     r = collect_fragments(
#         chunks,
#         _two_documents,
#     )
#     print(r)
#     assert (
#             r
#             == f"read page « {_P}foo{_S}(https://www.foo.org) »"
#     )


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
                    f"before {_P}1{_S}(id=99) after",
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
    _send(manage_references, f"{_P}NUMBER{_S}(id=1)") == "[1](a.html#chap1)"
    assert _send(manage_references, "") is None


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
                f"yes{_P}1{_S}(id=3), maybe{_P}2{_S}(id=2), no{_P}3{_S}(id=4), "
                f"yes{_P}4{_S}(id=1), error{_P}5{_S}(id=10)",
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
                f"yes{_P}1{_S}(id=3), maybe{_P}2{_S}(id=2), "
                f"no{_P}3{_S}(id=4), yes{_P}4{_S}(id=1), error{_P}5{_S}(id=10)",
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
                f"yes{_P}1{_S}(id=3), maybe{_P}2{_S}(id=2), "
                f"no{_P}3{_S}(id=4), yes{_P}4{_S}(id=1), error{_P}5{_S}(id=10)",
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
                f"yes{_P}1{_S}(id=3), maybe{_P}2{_S}(id=2), "
                f"no{_P}3{_S}(id=4), yes{_P}4{_S}(id=1), error{_P}5{_S}(id=10)",
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
                f"yes{_P}1{_S}(id=3), maybe{_P}2{_S}(id=2), "
                f"no{_P}3{_S}(id=4), yes{_P}4{_S}(id=1), error{_P}5{_S}(id=10)",
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
                f"yes{_P}1{_S}(id=3), maybe{_P}2{_S}(id=2), "
                f"no{_P}3{_S}(id=4), yes{_P}4{_S}(id=1), error{_P}5{_S}(id=10)",
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
                f"yes{_P}1{_S}(id=3), maybe{_P}2{_S}(id=2), "
                f"no{_P}3{_S}(id=4), yes{_P}4{_S}(id=1), error{_P}5{_S}(id=10)",
            )
            == "yes"
    )
    assert (
            _send(manage_references, "")
            == ', maybe<sup><a href="source2">1</a></sup>, no, yes<sup><a '
               'href="source1">2</a></sup>, error'
    )
    assert (
            _send(manage_references,
                  None) == '\n<ol><li><a href="source2">title2</a></li>'
                           '<li><a href="source1">title1</a></li></ol>'
    )

    # Test without title
    documents[0].metadata.pop("title")
    documents[1].metadata.pop("title")
    _send(manage_references, None)
    assert (
            _send(
                manage_references,
                f"yes{_P}1{_S}(id=3), maybe{_P}2{_S}(id=2), "
                f"no{_P}3{_S}(id=4), yes{_P}4{_S}(id=1), error{_P}5{_S}(id=10)",
            )
            == "yes"
    )
    assert (
            _send(manage_references, "")
            == ', maybe<sup><a href="source2">1</a></sup>, no, yes<sup><a '
               'href="source1">2</a></sup>, error'
    )
    assert (
            _send(manage_references,
                  None) == '\n<ol><li><a href="source2">source2</a></li>'
                           '<li><a href="source1">source1</a></li></ol>'
    )


def test_my_style() -> None:
    # Test My style, with the exclusion of big documents (total_pages > 5)
    def my_source(media: BaseMedia) -> str:
        if "row" in media.metadata:
            return f'{media.metadata["source"]}#{media.metadata["row"]}'
        return media.metadata["source"]

    class MyReferenceStyle(ReferenceStyle):
        source_id_key = my_source

        def format_reference(self, ref: int, media: BaseMedia) -> Optional[str]:
            get_total_pages = self._get_key_assigner(self.total_pages_key)
            total_pages = get_total_pages(media)
            if total_pages and total_pages > self.max_total_pages:
                return None
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
                f"yes{_P}1{_S}(id=3), maybe{_P}2{_S}(id=2), "
                f"no{_P}3{_S}(id=4), yes{_P}4{_S}(id=1), remove{_P}5{_S}(id=3), "
                f"error{_P}5{_S}(id=10)",
            )
            == "yes"
    )
    assert _send(manage_references, "") == (
        ", maybe[title2], no, yes[title1], remove, error"
    )
    assert _send(manage_references, None) == "\n\n- [1] source2#2\n- [2] source1#1\n"
