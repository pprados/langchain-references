"""
Tools to manage reference in a stream of tokens.

- Map the [ref](id=doc_id) to a new reference
- Use the same reference for different chunks from the same document
- Use strict order
- Add all references at the end of the stream
"""

from __future__ import annotations

import logging
import re
from abc import abstractmethod
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Generator,
    Iterable,
    Sequence,
    cast,
)

from langchain_core.documents.base import BaseMedia
from langchain_core.language_models import LanguageModelInput, LanguageModelOutput
from langchain_core.messages import AIMessageChunk, BaseMessage, BaseMessageChunk
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableGenerator,
    RunnablePassthrough,
)

logger = logging.getLogger(__name__)

# Use template similaire to OpenAI
_PREFIX = "【"  # Uses an unusual pattern
_SUFFIX = "†source】"
ALL_FORMAT_SUFFIX = "【†source】"

FORMAT_REFERENCES = (
    f"When referencing the documents, add a citation right after. "
    f'Use "{_PREFIX}ID_NUMBER{_SUFFIX}" for the citation '
    f'(e.g. "The Space Needle is in Seattle '
    f'{_PREFIX}1{_SUFFIX}{_PREFIX}2{_SUFFIX}.").'
)

# After this size, without reference, cancel the windows and wait a new '['
_MAX_WINDOWS_SIZE = 20

# Some LLM return a reference like [NUMBER](id=3)
# The number is not important, because we regenerate a new reference
_ids_pattern = re.compile(rf" *({re.escape(_PREFIX)}(\d+|NUMBER){re.escape(_SUFFIX)})")
_id_pattern = re.compile(rf"{re.escape(_PREFIX)}(\d+|NUMBER){re.escape(_SUFFIX)}")


# %% Different styles of references
class ReferenceStyle:
    source_id_key: str | Callable[[BaseMedia], str] = "source"
    """The metadata to identify the id of the parents """
    total_pages_key: str | Callable[[BaseMedia], str] = "total_pages"
    """The key with the total number of pages in the document"""
    max_total_pages: int = 4
    """The maximum number of pages to reference a document"""

    @staticmethod
    def _get_key_assigner(
        source_id_key: str | Callable[[BaseMedia], Any],
    ) -> Callable[[BaseMedia], Any]:
        """Get the source id from the document."""
        if isinstance(source_id_key, str):
            return lambda doc: doc.metadata.get(source_id_key)
        elif callable(source_id_key):
            if source_id_key.__self__:  # type: ignore
                return source_id_key.__func__  # type: ignore
            return source_id_key
        else:
            raise ValueError(
                f"source_id_key should be either None, a string or a callable. "
                f"Got {source_id_key} of type {type(source_id_key)}."
            )

    @abstractmethod
    def format_reference(self, ref: int, media: BaseMedia) -> str | None:
        """Format a reference in the text.
        :param ref: the reference number
        :param media: the document
        :return: the formatted reference or None to ignore the reference
        """
        ...

    @abstractmethod
    def format_all_references(self, refs: list[tuple[int, BaseMedia]]) -> str:
        """Format all references at the end of the text.
        :param refs: the list of references
        :return: the formatted list of references"""
        ...


class EmptyReferenceStyle(ReferenceStyle):
    """Empty style.
    Remove all references.
    """

    def format_reference(self, ref: int, media: BaseMedia) -> str | None:
        return ""

    def format_all_references(self, refs: list[tuple[int, BaseMedia]]) -> str:
        return ""


class TextReferenceStyle(ReferenceStyle):
    """Text only style.
    Remove all references.
    """

    def format_reference(self, ref: int, media: BaseMedia) -> str | None:
        return f"[{ref}]"

    def format_all_references(self, refs: list[tuple[int, BaseMedia]]) -> str:
        if not refs:
            return ""
        get_source = self._get_key_assigner(self.source_id_key)
        result = []
        for ref, media in refs:
            source = get_source(media)
            if media.metadata.get("title", ""):
                result.append(f"- [{ref}] {media.metadata['title']} ({source})\n")
            else:
                result.append(f"- [{ref}] {source}\n")
        if not result:
            return ""
        return "\n\n" + "".join(result)


class MarkdownReferenceStyle(ReferenceStyle):
    """Markdown style.
    Add reference in the bodie, with link to the source,
    and add a list of references at the end.
    """

    FOOT_NOTE: dict[str, str] = {
        "REF": "[^{ref}]",
        "TITLE_NOTE": "[^{ref}]: [{title}]({source})\n",
        "NOTE": "[^{ref}]: {source}\n",
    }
    NO_FOOT_NOTE: dict[str, str] = {
        "REF": '<a href="#fn{ref}" id="{ref}">[{ref}]</a></sup>',
        "TITLE_NOTE": '<sup id="fn{ref}" style="font-size: 0.7em;">{ref}.</a> '
        "[{title}]({source})</sup></small>  \n",
        "NOTE": '<sup id="fn{ref}" style="font-size: 0.7em;">{ref}.</a> '
        "{source}</sup>  \n",
    }

    def __init__(self, foot_note_compatibe: bool = True):
        if foot_note_compatibe:
            self._compatible = MarkdownReferenceStyle.FOOT_NOTE
        else:
            self._compatible = MarkdownReferenceStyle.NO_FOOT_NOTE

    # _TEMPLATE_NOTE="[^{ref}]: {source}\n"
    def format_reference(self, ref: int, media: BaseMedia) -> str:
        source = self._get_key_assigner(self.source_id_key)(media)
        # return f"<sup>[[{ref}]({source})]</sup>"
        return self._compatible["REF"].format(ref=ref, source=source)

    def format_all_references(self, refs: list[tuple[int, BaseMedia]]) -> str:
        if not refs:
            return ""
        get_source = self._get_key_assigner(self.source_id_key)
        result = []
        for ref, media in refs:
            source = get_source(media)
            if media.metadata.get("title", ""):
                result.append(
                    self._compatible["TITLE_NOTE"].format(
                        ref=ref, title=media.metadata["title"], source=source
                    )
                )
            else:
                result.append(self._compatible["NOTE"].format(ref=ref, source=source))
        if not result:
            return ""
        return "\n\n" + "".join(result)


class HTMLReferenceStyle(ReferenceStyle):
    """HTML style.
    Add reference in the bodie, with link to the source,
    and add a list of references at the end.
    """

    def format_reference(self, ref: int, media: BaseMedia) -> str:
        source = self._get_key_assigner(self.source_id_key)(media)
        return f'<sup><a href="{source}">{ref}</a></sup>'

    def format_all_references(self, refs: list[tuple[int, BaseMedia]]) -> str:
        get_source = self._get_key_assigner(self.source_id_key)
        if not refs:
            return ""
        result = ["\n<ol>"]
        for _, media in refs:
            source = get_source(media)
            if media.metadata.get("title", ""):
                result.append(
                    f'<li><a href="{source}">{media.metadata["title"]}</a></li>'
                )
            else:
                result.append(f'<li><a href="{source}">{source}</a></li>')
        result.append("</ol>")
        if not result:
            return ""
        return "".join(result)


def _analyse_doc_ids(
    style: ReferenceStyle,
    mediums: Sequence[BaseMedia],
) -> dict[int, int]:
    # For each doc, find the id of referenced document
    source_id_key_get = ReferenceStyle._get_key_assigner(style.source_id_key)
    seen: dict[str, int] = {}
    uniq_id_for_chunk: dict[int, int] = {}
    gen_id = 1
    for id, media in enumerate(mediums):
        key = source_id_key_get(media)
        if key in seen:
            uniq_id_for_chunk[id] = seen[key]
        else:
            seen[key] = gen_id
            uniq_id_for_chunk[id] = gen_id
            gen_id += 1
    return uniq_id_for_chunk


# %%
def _patch_id(
    style: ReferenceStyle,
    medium: Sequence[BaseMedia],
) -> Generator[tuple[str, int] | dict[int, BaseMedia] | None, str | None, None]:
    # Calculate a uniq id for each chunk
    # in order to allow the injection of a single reference, in order.
    uniq_id_for_chunk = _analyse_doc_ids(style, medium)
    ids: dict[int, BaseMedia] = {}
    last_ref = 0
    new_reference_number: dict[int, int] = {}
    formated_result: tuple[str, int] = ("", 0)
    while True:
        last = 0
        result = ""
        content = yield (formated_result)
        if content is None:
            break
        for m in _ids_pattern.finditer(content):
            # Add the text before the reference
            result += content[last : m.start()]
            llm_reference = content[m.start() : m.end()]
            matched = _id_pattern.search(llm_reference)
            assert matched, "a previous test must ensure that the pattern is correct"
            if matched[1] == "NUMBER":
                continue  # Ignore reference
            llm_index = int(matched[1]) - 1  # llm doc position
            if (llm_index < 0) or (llm_index >= len(medium)):
                logger.warning(f"LLM return an invalid document reference {llm_index}.")
                last = m.end()
                continue  # Ignore invalid references
            media = medium[llm_index]  # The corresponding document
            # A uniq id associated to the root document
            uniq_id = uniq_id_for_chunk[llm_index]
            if uniq_id not in new_reference_number:
                # It's the first time I refer to this document ?
                last_ref += 1
                new_reference_number[uniq_id] = last_ref
                new_ref = last_ref
            else:
                # I already refer to this document, use the same reference
                new_ref = new_reference_number[uniq_id]
            # FIXME: reference another doc, with the same URL
            # if (new_ref in ids) and ids[new_ref] != media:
            #     logger.warning(f"LLM generated the same reference [{llm_ref}] twice.")
            #     last = m.end()
            #     continue  # Ignore invalid references
            # Ask to format the reference (id and link or whatever)
            new_reference = style.format_reference(new_ref, media)
            if new_reference is not None:
                # Save the reference
                ids[new_ref] = media
                # Add the new formatted reference
                result += new_reference
            else:
                last_ref -= 1
            last = m.end()
        # Return all results and the last position
        formated_result = (result, last)
    yield ids


def _manage_references(
    *,
    style: ReferenceStyle,
    medium: Sequence[BaseMedia],
) -> Generator[BaseMessage | None, AIMessageChunk | None, None]:
    """Manage ids and references in the stream."""
    # State machine to detect in a stream, [1](id=n)
    while True:
        patch_id = _patch_id(style=style, medium=medium)
        patch_id.send(None)
        wait = True
        windows_str = ""
        result: AIMessageChunk | None = None
        windows_patched: str
        matched = None
        text_fragment = ""
        message: str | BaseMessageChunk | None = None
        while True:
            message = yield result
            result = None
            if isinstance(message, BaseMessageChunk):
                if isinstance(message.content, str):
                    text_fragment = message.content
                else:
                    raise ValueError(f"Invalid content type {type(message.content)}")
            if message is None:
                text_fragment = ""
            if isinstance(message, str):
                text_fragment = message
            if wait:
                windows_str += text_fragment
                if _PREFIX in cast(str, windows_str):
                    pos = windows_str.find(_PREFIX)
                    before = windows_str[:pos]
                    after = windows_str[pos:]
                    windows_str = after
                    wait = False

                    if before:
                        result = AIMessageChunk(content=before)
                    else:
                        matched = _ids_pattern.search(windows_str)
                else:
                    result = None
                    if text_fragment:
                        if len(windows_str) > len(_PREFIX):
                            result = AIMessageChunk(
                                content=windows_str[: -len(_PREFIX)]
                            )
                            windows_str = windows_str[-len(_PREFIX) :]
            else:
                windows_str += text_fragment
                matched = _ids_pattern.search(windows_str)
                if not matched:
                    if len(windows_str) > _MAX_WINDOWS_SIZE:
                        # Find prefix without reference
                        # Try to return to wait state
                        pos = windows_str.find(_PREFIX, len(_PREFIX))
                        if pos > 0:
                            wait = True
                            result = AIMessageChunk(content=windows_str[:pos])
                            windows_str = windows_str[pos:]
                        else:
                            yield AIMessageChunk(
                                content=windows_str[:_MAX_WINDOWS_SIZE]
                            )
                            windows_str = windows_str[_MAX_WINDOWS_SIZE:]
                            result = None
            if matched:
                windows_patched, last = cast(
                    tuple[str, int], patch_id.send(windows_str)
                )
                windows_str = windows_str[last:]
                wait = _PREFIX not in windows_str
                result = AIMessageChunk(content=windows_patched)
            matched = None
            if message is None:
                break
        ids = cast(dict[int, BaseMedia], patch_id.send(None))
        yield AIMessageChunk(
            content=windows_str
            + style.format_all_references(
                cast(list[tuple[int, BaseMedia]], ids.items())
            )
        )


# %% Lambdas for langchain
def _update_references(
    inputs: Iterable[dict[str, Any]],
    runnable: Runnable,
    style: ReferenceStyle,
    config: RunnableConfig | None = None,
) -> Iterable[BaseMessage | None]:
    for input in inputs:
        if "input" not in input:
            continue
        assert isinstance(
            input, dict
        ), "The input to manage_references() must be a dict."
        medium = input["input"]["medium"]
        manage_references = _manage_references(style=style, medium=medium)
        manage_references.send(None)  # Start
        chunk = None
        for token in runnable.stream(input["input"], config=config):
            # Inject the token in the FSM*
            chunk = manage_references.send(token)
            if chunk:
                yield chunk
        while chunk:
            chunk = manage_references.send(AIMessageChunk(content=""))
            if chunk:
                yield chunk
        chunk = manage_references.send(None)  # Stop
        if chunk:
            yield chunk


async def _aupdate_references(
    inputs: AsyncIterable[dict[str, Any]],
    runnable: Runnable,
    style: ReferenceStyle,
    config: RunnableConfig | None = None,
) -> AsyncIterable[BaseMessage | None]:
    async for input in inputs:
        if "input" not in input:
            continue
        medium = input["input"]["medium"]
        manage_references = _manage_references(style=style, medium=medium)
        manage_references.send(None)  # Start
        for token in runnable.stream(input["input"], config=config):
            # Inject the token in the FSM*
            x = manage_references.send(token)
            if x:
                yield x
        yield manage_references.send(None)  # Stop


# %% Chain factory
def manage_references(
    runnable: Runnable[LanguageModelInput, LanguageModelOutput],
    *,
    documents_key: str = "documents",
    style: ReferenceStyle = MarkdownReferenceStyle(foot_note_compatibe=False),
) -> Runnable[LanguageModelInput, LanguageModelOutput]:
    return (
        RunnablePassthrough.assign(medium=lambda x: x[documents_key])
        | RunnablePassthrough.assign(input=lambda x: x)  # Duplicate all the input
        | RunnableGenerator(
            transform=_update_references, atransform=_aupdate_references  # type: ignore
        ).bind(runnable=runnable, style=style)
    )


# %% Runnable implementation

# class ManageReferences(RunnableSerializable[LanguageModelInput, Runnable[
#     LanguageModelInput, LanguageModelOutput]]):
#     _r: Runnable = PrivateAttr()
#
#     def __init__(self, runnable: Runnable[LanguageModelInput, LanguageModelOutput],
#                  *,
#                  documents_key: str = "documents",
#                  style: ReferenceStyle = MarkdownReferenceStyle()) -> None:
#         super().__init__()
#         self._r = manage_references(runnable,
#         documents_key=documents_key, style=style)
#
#     def invoke(self, input: LanguageModelOutput,
#                config: RunnableConfig | None = None) -> Runnable[
#         LanguageModelInput, LanguageModelOutput]:
#         return self._r.invoke(input, config=config)
#
#     def stream(
#             self,
#             input: LanguageModelInput,
#             config: RunnableConfig | None = None,
#             **kwargs: Any | None,
#     ) -> Iterator[LanguageModelOutput]:
#         return self._r.stream(input, config=config)
