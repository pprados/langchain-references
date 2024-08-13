Langchain-Reference
===================

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/pprados/langchain-references?quickstart=1)

**"Ask a question, get a comprehensive answer and directly access the sources 
used to develop that answer."**

It's a very difficult goal to achieve.

**Now that's not a problem!**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pprados/langchain-references/blob/master/langchain_reference.ipynb)

## Introduction
When publishing the LLM's response, it can be helpful to include links to the documents 
used to produce the answer. This way, the user can learn more or verify that the 
response is accurate.

The list of documents is retrieved from the vector database. Each fragment carries 
metadata that allows for precise identification of its origin (the URL, the page, 
the title, the position of the first character, etc.).

To be clearer, let's position ourselves in a typical scenario. A question is posed in 
a prompt, which is then processed with five documents:

- `a.html#chap1`: The fragment relates to Chapter 1 of the `a.html` file.
- `a.html#chap2`: The fragment relates to Chapter 2 of the `a.html` file.
- `b.pdf`: The fragment comes from the `b.pdf` file.
- `b.pdf`: Another fragment also comes from the `b.pdf` file.
- `c.csv`: The fragment is a line from the `c.csv` file.

Given the scenario where the LLM uses multiple fragments from different documents to 
generate a response, the references should be formatted as footnotes, reflecting 
the different sources. For example, the LLM answers several questions using the 
referenced fragments:
```markdown
Yes[1](id=3), certainly[2](id=2), no[3](id=4), yes[4](id=1)
```
In this situation, the first four fragments are used, but not the last one. 
The first two have different URLs, even though they come from the same document. 
The next two share the same URL but refer to different fragments.

The naive approach is to list all the injected documents after the response and, 
if possible, extract a specific title for each fragment.  
```markdown
Yes, certainly, no, yes

- [a chap1](a.html#chap1)
- [a chap2](a.html#chap2)
- [b frag1](b.pdf)
- [b frag2](b.pdf)
- [c](c.csv)
```
---
Yes, certainly, no, yes

- [a chap1](a.html#chap1)
- [a chap2](a.html#chap2)
- [b frag1](b.pdf)
- [b frag2](b.pdf)
- [c](c.csv)
---

Optionally, duplicates can be filtered out.

We observe that the result is not satisfactory. First, the user will be disappointed 
when reading the file `c.csv` to find that it doesn’t contain any information 
supporting the response. This file should be excluded from the reference list since 
it provides no useful information and was not used by the LLM to generate the answer. 
There are also two different links leading to the same document, which could confuse 
the user as to why this is the case.

What should be produced is closer to this:
```markdown
Yes[1], certainly[2], no[3], yes[4]

- [1],[3] [b](b.pdf)
- [2]     [a chap2](a.html#chap2)
- [4]     [a chap1](a.html#chap1)
```
---
Yes[1], certainly[2], no[3], yes[4]

- [1],[3] [b](b.pdf)
- [2]     [a chap2](a.html#chap2)
- [4]     [a chap1](a.html#chap1)
---
We identify fragments sharing the same URL to combine reference numbers and avoid 
unreferenced documents.

The best solution is to adjust the reference numbers when they share the same URL. 
This adjustment should be made during the LLM’s response generation to achieve the 
following:
```markdown
yes[1], certainly[2], no[1], yes[3]

- [1] [b](b.pdf)
- [2] [a chap2](a.html#chap2)
- [3] [a chap1](a.html#chap1)
```
---
yes[1], certainly[2], no[1], yes[3]

- [1] [b](b.pdf)
- [2] [a chap2](a.html#chap2)
- [3] [a chap1](a.html#chap1)
---

Note that the reference numbers have been adjusted. As a result, 
we have a reference list that resembles what a human would have created.

**This complexity cannot reasonably be delegated to the LLM.** It would require providing 
the URLs for each fragment and crafting a prompt in the hope that it would always 
calculate correctly. Since handling URLs is not its strength, it’s better to relieve 
it of this responsibility and implement deterministic code capable of consistently 
performing the necessary calculations and adjustments. In the process, links can be 
directly embedded in the references.

```markdown
yes<sup>[[1](b.pdf)]</sup>, certainly<sup>[[2](a.html#chap2)]</sup>, 
no<sup>[[1](b.pdf)]</sup>, yes<sup>[[3](a.html#chap1)]</sup>

- [1] [b](b.pdf)
- [2] [a chap2](a.html#chap2)
- [3] [a chap1](a.html#chap1)
```
---
yes<sup>[[1](b.pdf)]</sup>, certainly<sup>[[2](a.html#chap2)]</sup>, 
no<sup>[[1](b.pdf)]</sup>, yes<sup>[[3](a.html#chap1)]</sup>

- [1] [b](b.pdf)
- [2] [a chap2](a.html#chap2)
- [3] [a chap1](a.html#chap1)
---

## Usage
**You can't ask too much of an LLM.** Imperative code is often the best solution. 
To manage document references correctly, we'll separate responsibilities into two 
parts. The first is not too complex for an LLM: indicate a reference number, 
followed by the identifier of the fragment from which the answer is extracted. The 
second part is the responsibility of a Python code: adjusting reference numbers if 
there are duplicates, injecting URLs to the original documents if necessary, then 
concluding the prompt with the list of all references.

Having the list of documents that have been injected into the prompt, it is possible 
to add an identifier (the position of each document in the list), so that LLM can 
respond with the unique number of the injected document. In this way, it is possible 
to retrieve each original document and use the metadata to build a URL, for example. 
The following prompt asks LLM to handle references simply, in the form : 
`[<num_reference>](id=<position_du_fragment>)`.

```python
from langchain_references import FORMAT_REFERENCES
print(f{FORMAT_REFERENCES=})
```
```text
FORMAT_REFERENCES='When referencing the documents, add a citation right after.' 
'Use "[NUMBER](id=ID_NUMBER)" for the citation (e.g. "The Space Needle is in '
'Seattle [1](id=55)[2](id=12).").'
```
And the prompt:
```python
prompt=ChatPromptTemplate.from_template(
"""
Here, the context: 
{documents}

{format_references}

Question : {question}
""")
```
The context must be built up by adding a reference to each document.
```python
def format_docs(docs):
    return "\n".join(
        # Add a document id so that LLM can reference it 
        [f"<document id={i+1}>\n{doc.page_content}\n</document>\n" 
         for i,doc in enumerate(docs)]
    )
```
Then, thanks to `langchain-references`, to modify the tokens produced by the LLM.
Encapsulate the invocation of the model with `manage_references()` to adjust the
reference numbers and inject the URLs of the original documents.
```python
from langchain_references import manage_references
chain = manage_references(
    context
    | rag_prompt
    | model,
) | StrOutputParser()
```
Now, invoke the chain with the documents and the question.
```python
question = "What are the approaches to Task Decomposition?"

docs = vectorstore.similarity_search(question)

# Run
print(chain.invoke({"documents": docs, "question": question}))
```
The response from the LLM will be:
```text
The difference subject of mathematics can refer to various areas within the field, 
such as number theory, algebra, geometry, analysis, and set theory. Each area 
focuses on different concepts, methods, and theorems relevant to both mathematics 
and empirical sciences [1](id=1). Additionally, mathematical games and puzzles 
highlight the distinction in engagement and required expertise within the mathematical 
domain [3](id=3).
```

The response will be:
```markdown
The difference subject of mathematics can refer to various areas within the field, 
such as number theory, algebra, geometry, analysis, and set theory. Each area 
focuses on different concepts, methods, and theorems relevant to both mathematics 
and empirical sciences <sup>[[1](https://en.wikipedia.org/wiki/Mathematics)]</sup>. Additionally, mathematical games and puzzles 
highlight the distinction in engagement and required expertise within the mathematical 
domain <sup>[[3](https://en.wikipedia.org/wiki/Mathematical_game)]</sup>.

- **1** [Mathematics](https://en.wikipedia.org/wiki/Mathematics)
- **2** [Mathematical game](https://en.wikipedia.org/wiki/Mathematical_game)
```
---
The difference subject of mathematics can refer to various areas within the field, 
such as number theory, algebra, geometry, analysis, and set theory. Each area 
focuses on different concepts, methods, and theorems relevant to both mathematics 
and empirical sciences <sup>[[1](https://en.wikipedia.org/wiki/Mathematics)]</sup>. Additionally, mathematical games and puzzles 
highlight the distinction in engagement and required expertise within the mathematical 
domain <sup>[[3](https://en.wikipedia.org/wiki/Mathematical_game)]</sup>.

- **1** [Mathematics](https://en.wikipedia.org/wiki/Mathematics)
- **2** [Mathematical game](https://en.wikipedia.org/wiki/Mathematical_game)
---

The `manage_references()` take a `Runnable[LanguageModelInput, LanguageModelOutput]` as 
an argument and return a `Runnable[LanguageModelInput, LanguageModelOutput]`.

The input of this `Runnable` must a dictionary with the keys *documents_key* 
(`documents` by default) and what the runnable wants as a parameter.

## Style
Different styles can be used to display the references. The default style is:
Markdown, but you can use:
- `EmptyReferenceStyle` : no references are produce
- `TextReferenceStyle` : for console output
- `MarkdownReferenceStyle` : format markdown output
- `HTMLReferenceStyle` : format html output

You can adjust the style to suit the specific requirements of your documents.
```python
from langchain_references import ReferenceStyle
from langchain_core.documents.base import BaseMedia
from typing import List, Tuple
class MyReferenceStyle(ReferenceStyle):
    source_id_key = lambda \
        media: f'{media.metadata["source"]}#{media.metadata["row"]}'

    def format_reference(self, ref: int, media: BaseMedia) -> str:
        return f" (See {media.metadata['title']})"

    def format_all_references(self, refs: List[Tuple[int, BaseMedia]]) -> str:
        if not refs:
            return ""
        result = []
        for ref, media in refs:
            source = self.source_id_key.__func__(media)
            result.append(f"- [{ref}] {source}\n")
        if not result:
            return ""
        return "\n\n" + "".join(result)
```

## How does it work?
On the fly, each token is captured to identify the pattern of references. As soon as 
the beginning of a text seems to match, tokens are accumulated until references are 
identified or the capture is abandoned, as this is a false alarm. The accumulated 
tokens are then produced, before the analysis is resumed.
As soon as a token appears, it is assigned an identifier, in relation to the various 
documents present. Then `format_reference()` is invoked. 
When there are no more tokens, the list of documents used for the response is 
constructed and added as the final fragment, via `format_all_references()`.