Langchain-Reference
===================

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/pprados/langchain-references?quickstart=1)

**"Ask a question, get a comprehensive answer and directly access the sources 
used to develop that answer."**

It's a very difficult goal to achieve.

**Now that's not a problem!**. In just one hour, you can significantly improve your RAG 
project. Do you want an answer like this?

---
Mathematical games are structured activities defined by clear mathematical parameters, 
focusing on strategy and skills without requiring deep mathematical knowledge, such as 
tic-tac-toe or chess <sup>[[1](https://en.wikipedia.org/wiki/Mathematics)]</sup>. In contrast, mathematics competitions, like 
the International Mathematical Olympiad, involve participants solving complex 
mathematical  problems, often requiring proof or detailed solutions 
<sup>[[2](https://en.wikipedia.org/wiki/Mathematical_game)]</sup>. Essentially, games are for enjoyment and skill development, 
while competitions test and challenge mathematical understanding and problem-solving 
abilities..

- **1** [Mathematics](https://en.wikipedia.org/wiki/Mathematics)
- **2** [Mathematical game](https://en.wikipedia.org/wiki/Mathematical_game)
---


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pprados/langchain-references/blob/master/langchain_reference.ipynb)

## Introduction
With a RAG project, when publishing the LLM's response, it can be helpful to include 
links to the documents used to produce the answer. This way, the user can learn more 
or verify that the response is accurate.

The list of documents is retrieved from the vector database. Each fragment carries 
metadata that allows for precise identification of its origin (the URL, the page, 
the title, the position of the first character, etc.).

![Documents/Chunks/Facts](documents_chunk_fact.png)
The different combinations, document, chunk and fact (argument allowing an answer) lead 
to significant complexity in the creation of the list of documents.
- The answer is produced from a fact, extracted from a fragment coming from a single document. 
The link to the fragment is sufficient. The link can reference the document itself, if 
it is not too long. Otherwise, it provides no help to the user. He cannot reread 200 
pages to find the fact.
- The answer is produced from several facts, extracted from several fragments of several 
documents. In this case, links to both fragments must be produced.
- The response is produced from several facts, extracted from several fragments of the 
same document. If there are links for each fragment, then both links must be produced. 
If there are only links to the original document, you should not indicate 2 documents, 
even though they are the same. Footnote reference numbers need to be adjusted.
- The answer is produced from several facts, extracted from a single fragment, from a 
single document. In this case, the footnote reference numbers must be adjusted to 
have the same number.
- A fragment does not carry a fact that was used for the answer. The document 
should not be referenced.


To be clearer, let's position ourselves in a typical scenario. A question is posed in 
a prompt, which is then processed with five documents:

1. `a.html#chap1`: The fragment relates to Chapter 1 of the `a.html` file.
2. `a.html#chap2`: The fragment relates to Chapter 2 of the `a.html` file.
3. `b.pdf`: The fragment comes from the `b.pdf` file.
4. `b.pdf`: Another fragment also comes from the `b.pdf` file.
5. `c.pdf`: A fragment from a big document `c.pdf`, with 200 pages.
6. `d.csv`: The fragment is a line from the `c.csv` file.

Given the scenario where the LLM uses multiple fragments from different documents to 
generate a response, the references should be formatted as footnotes, reflecting 
the different sources. For example, the LLM answers several questions using the 
referenced fragments:
```markdown
Yes[1](id=3), certainly[2](id=2), no[3](id=4), yes[4](id=1), yes[5](id=5)
```
In this situation, the first five fragments are used, but not the last one. 
The first two have different URLs, even though they come from the same document. 
The next two share the same URL but refer to different fragments. The fifth comes from 
a 200-page document, which doesn't help the user.

The naive approach is to list all the injected documents after the response and, 
if possible, extract a specific title for each fragment.  
```markdown
Yes, certainly, no, yes

1. [a chap1](a.html#chap1)
2. [a chap2](a.html#chap2)
3. [b frag1](b.pdf)
4. [b frag2](b.pdf)
5. [c](c.pdf)
6. [d](d.csv)
```
---
Yes, certainly, no, yes

1. [a chap1](a.html#chap1)
2. [a chap2](a.html#chap2)
3. [b frag1](b.pdf)
4. [b frag2](b.pdf)
5. [c](c.pdf)
6. [d](d.csv)
---

Optionally, duplicates can be filtered out. The most absurd situation in this scenario 
is when the LLM says he doesn't know the answer, but proposes a list 
of documents anyway. Other scenarios are problematic with this approach. If the answer 
is based on several documents, it is not possible to associate each argument with a 
particular document. If several documents are used to answer the question, it's not 
necessary to list them all - just one is enough. All this creates confusion and is not 
at the level of quality expected by the user.

We observe that the result is not satisfactory. First, the user will be disappointed, 
when reading the `c.pdf` file, to discover that the document is 200 pages long and 
that `d.csv` has no facts to justify the answer. File `d.csv` should be excluded from 
the list of references, as it provides no useful information. It is not used by 
LLM to produce the answer. 

What should be produced is closer to this:
```markdown
Yes<sup>[1]</sup>, certainly<sup>[2]</sup>, no<sup>[3]</sup>, yes<sup>[4]</sup>, yes<sup>[5]</sup>

- [1],[3] [b](b.pdf)
- [2]     [a chap2](a.html#chap2)
- [4]     [a chap1](a.html#chap1)
- [5]     [c](c.pdf)
```
---
Yes<sup>[1]</sup>, certainly<sup>[2]</sup>, no<sup>[3]</sup>, yes<sup>[4]</sup>, yes<sup>[5]</sup>

- [1],[3] [b](b.pdf)
- [2]     [a chap2](a.html#chap2)
- [4]     [a chap1](a.html#chap1)
- [5]     [c](c.pdf)
---
We identify fragments sharing the same URL to combine reference numbers and avoid 
unreferenced documents.

The best solution is to adjust the reference numbers when they share the same URL. 
This adjustment should be made during the LLM’s response generation to achieve the 
following:
```markdown
Yes<sup>[1]</sup>, certainly<sup>[2]</sup>, no<sup>[1]</sup>, yes<sup>[3]</sup>, yes<sup>[4]</sup>

- [1] [b](b.pdf)
- [2] [a chap2](a.html#chap2)
- [3] [a chap1](a.html#chap1)
- [4] [c](c.pdf)
```
---
Yes<sup>[1]</sup>, certainly<sup>[2]</sup>, no<sup>[1]</sup>, yes<sup>[3]</sup>, yes<sup>[4]</sup>

- [1] [b](b.pdf)
- [2] [a chap2](a.html#chap2)
- [3] [a chap1](a.html#chap1)
- [4] [c](c.pdf)
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
no<sup>[[1](b.pdf)]</sup>, yes<sup>[[3](a.html#chap1)]</sup>, yes<sup>[[4](...)]</sup>

- [1] [b](b.pdf)
- [2] [a chap2](a.html#chap2)
- [3] [a chap1](a.html#chap1)
- [4] [c](c.pdf)
```
---
yes<sup>[[1](b.pdf)]</sup>, certainly<sup>[[2](a.html#chap2)]</sup>, 
no<sup>[[1](b.pdf)]</sup>, yes<sup>[[3](a.html#chap1)]</sup>, yes<sup>[[4](...)]</sup>

- [1] [b](b.pdf)
- [2] [a chap2](a.html#chap2)
- [3] [a chap1](a.html#chap1)
- [4] [c](c.pdf)
---

## Usage:
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
`[<number_of_reference>](id=<index_of_fragment>)`.

### Chain without retriever
To use a chain without a retriever, you need to apply some similar steps.

Get the format of the references.
```python
from langchain_references import FORMAT_REFERENCES
print(f{FORMAT_REFERENCES=})
```
```text
FORMAT_REFERENCES='When referencing the documents, add a citation right after.' 
'Use "[NUMBER](id=ID_NUMBER)" for the citation (e.g. "The Space Needle is in '
'Seattle [1](id=55)[2](id=12).").'
```
And create a prompt:
```python
prompt=rag_prompt = ChatPromptTemplate.from_template(
    """
You are an assistant for question-answering tasks. Use the following pieces of retrieved documents to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

{format_references}

<documents>
{context}
</documents>

Answer the following question:

{question}""",
    partial_variables={"format_references":FORMAT_REFERENCES})
```
The context must be built up by adding a reference to each document.
```python
def format_docs(docs):
    return "\n".join(
        # Add a document id so that LLM can reference it 
        [f"<document id={i+1}>\n{doc.page_content}\n</document>\n" 
         for i,doc in enumerate(docs)]
    )
context = RunnablePassthrough.assign(
    context=lambda input: format_docs(input["documents"]),
)
```
Then, thanks to `langchain-references`, to modify the tokens produced by the LLM.
Encapsulate the invocation of the model with `manage_references()` to adjust the
reference numbers and inject the URLs of the original documents.
```python
from langchain_references import manage_references
chain = context | manage_references(rag_prompt| model) | StrOutputParser()
```
Now, invoke the chain with the documents and the question.
```python
question = "What is the difference kind of games and competition of mathematics?"

docs = vectorstore.similarity_search(question)

# Run
print(chain.invoke({"documents": docs, "question": question}))
```
The response from the LLM will be:
```text
Mathematical games are structured activities defined by clear mathematical parameters, 
focusing on strategy and skills without requiring deep mathematical knowledge, such as 
tic-tac-toe or chess [1](id=1). In contrast, mathematics competitions, like the 
International Mathematical Olympiad, involve participants solving complex mathematical 
problems, often requiring proof or detailed solutions [2](id=2). Essentially, games 
are for enjoyment and skill development, while competitions test and challenge 
mathematical understanding and problem-solving abilities.'
```

The response with `manage_reference()` will be:
```markdown
Mathematical games are structured activities defined by clear mathematical parameters, 
focusing on strategy and skills without requiring deep mathematical knowledge, such as 
tic-tac-toe or chess <sup>[[1](https://en.wikipedia.org/wiki/Mathematics)]</sup>. In contrast, mathematics competitions, like 
the International Mathematical Olympiad, involve participants solving complex 
mathematical  problems, often requiring proof or detailed solutions 
<sup>[[2](https://en.wikipedia.org/wiki/Mathematical_game)]</sup>. Essentially, games are for enjoyment and skill development, 
while competitions test and challenge mathematical understanding and problem-solving 
abilities..

- **1** [Mathematics](https://en.wikipedia.org/wiki/Mathematics)
- **2** [Mathematical game](https://en.wikipedia.org/wiki/Mathematical_game)
```
---
Mathematical games are structured activities defined by clear mathematical parameters, 
focusing on strategy and skills without requiring deep mathematical knowledge, such as 
tic-tac-toe or chess <sup>[[1](https://en.wikipedia.org/wiki/Mathematics)]</sup>. In contrast, mathematics competitions, like 
the International Mathematical Olympiad, involve participants solving complex 
mathematical  problems, often requiring proof or detailed solutions 
<sup>[[2](https://en.wikipedia.org/wiki/Mathematical_game)]</sup>. Essentially, games are for enjoyment and skill development, 
while competitions test and challenge mathematical understanding and problem-solving 
abilities..

- **1** [Mathematics](https://en.wikipedia.org/wiki/Mathematics)
- **2** [Mathematical game](https://en.wikipedia.org/wiki/Mathematical_game)
---

The `manage_references()` take a `Runnable[LanguageModelInput, LanguageModelOutput]` as 
an argument and return a `Runnable[LanguageModelInput, LanguageModelOutput]`.

The input for this `Runnable` must be a dictionary with the keys *documents_key* 
(default is `documents`) and whatever the runnable requires as a parameter.

### Chain with retriever
With a chain with a retriever, the process is the same, but the documents are retrieved
by the retriever. The retriever must be added to the chain before the LLM.

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
context = (
    RunnableParallel(
        # Get list of documents, necessary for reference analysis
        documents=(itemgetter("question") | retriever),
        # and question
        question=itemgetter("question"),
    ).assign(
        context=lambda input: format_docs(input["documents"]),
        format_references=lambda _: FORMAT_REFERENCES,
    )
)
```
And invoke with:
```python
answer = (context | manage_references(rag_prompt | model) | StrOutputParser() ).invoke({"question": question})
pprint(answer)
```
## Style
Different styles can be used to display the references. The default style is:
Markdown, but you can use:
- `EmptyReferenceStyle` : no references are produced
- `TextReferenceStyle` : for console output
- `MarkdownReferenceStyle` : format markdown output
- `HTMLReferenceStyle` : format html output

You can use a class derived from `EmptyReferenceStyle` to trace the fragments used in 
the response, without informing the user. This is useful for analyzing unsatisfactory 
responses.

You can adjust the style or how to retrieve the URL from the metadata.
```python
from langchain_references import ReferenceStyle
from langchain_core.documents.base import BaseMedia
from typing import List, Tuple
def my_source_id(media: BaseMedia) -> str:
    # Documents loaded from a CSV file with 'row' in metadata
    return f'{media.metadata["source"]}#row={media.metadata["row"]}'

class MyReferenceStyle(ReferenceStyle):
    source_id_key = my_source_id

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
```
```python
chain = (context 
         | manage_references(rag_prompt| model, style=MyReferenceStyle()) 
         | StrOutputParser())
```

## How does it work?
On the fly, each token is captured to identify the pattern of references. As soon as 
the beginning of a text seems to match, tokens are accumulated until references are 
identified or the capture is abandoned, as this is a false detection. The accumulated 
tokens are then produced, before the analysis is resumed.
As soon as a token appears, it is assigned an identifier, in relation to the various 
documents present. Then `format_reference()` is invoked. 
When there are no more tokens, the list of documents used for the response is 
constructed and added as the final fragment, via `format_all_references()`.