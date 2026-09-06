from typing import Iterator, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided documents.
Answer concisely and accurately using ONLY the context provided.
If the answer is not in the context, say "I don't have enough information to answer that."
Always cite the source document name when referencing information.

Context: {context}"""

CONTEXTUALIZE_PROMPT = """Given a chat history and the latest user question which might reference
context in the chat history, formulate a standalone question which can be understood
without the chat history. Do NOT answer the question, just reformulate it if needed.
Otherwise, return the question as-is."""


def create_llm(provider: str, model_name: str, api_key: Optional[str] = None) -> BaseChatModel:
    provider = provider.lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model_name, temperature=0, api_key=api_key)

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, temperature=0, api_key=api_key)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, temperature=0, api_key=api_key)

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_name, temperature=0)

    else:
        raise ValueError(f"Unsupported provider: {provider}. Use groq, openai, anthropic, or ollama.")


class QAChain:
    """Retrieval + LLM chain supporting batch (``invoke``) and streaming
    (``stream``) execution. The retrieved source documents for the most recent
    call are exposed on ``last_context`` so the UI can render citations.

    Retrieval is store-agnostic: it uses ``similarity_search`` (works on both
    local Chroma and Supabase pgvector) and optionally restricts results to a
    list of source file names. Because Supabase's filter is exact-match on
    metadata (``metadata @> filter``), multi-source selection runs one query per
    source and merges the results.
    """

    def __init__(self, llm, vectorstore, contextualize_chain, qa_prompt, k, doc_sources=None):
        self.llm = llm
        self.vectorstore = vectorstore
        self.contextualize_chain = contextualize_chain
        self.qa_prompt = qa_prompt
        self.k = k
        self.doc_sources = doc_sources or []
        self.is_chroma = isinstance(vectorstore, Chroma)
        self.last_context: List[Document] = []

    def _standalone_question(self, inputs: dict) -> str:
        if inputs.get("chat_history"):
            return self.contextualize_chain.invoke(inputs)
        return inputs["input"]

    def _retrieve(self, question: str) -> List[Document]:
        if self.doc_sources:
            if self.is_chroma:
                docs = self.vectorstore.similarity_search(
                    question,
                    k=self.k * 3,
                    filter={"source": {"$in": self.doc_sources}},
                )
            else:
                # Exact-match filter per source, then merge.
                per_source_k = max(3, self.k)
                docs = []
                for source in self.doc_sources:
                    try:
                        docs.extend(self.vectorstore.similarity_search(
                            question,
                            k=per_source_k,
                            filter={"source": source},
                        ))
                    except Exception as e:
                        print(f"Warning: retrieval failed for {source}: {e}")
        else:
            docs = self.vectorstore.similarity_search(question, k=self.k * 3)

        seen = set()
        unique: List[Document] = []
        for d in docs:
            # Dedup on source + page + content prefix so a chunk from the same
            # source/page is not returned twice.
            key = (
                d.metadata.get("source"),
                d.metadata.get("page"),
                d.page_content[:200],
            )
            if key not in seen:
                seen.add(key)
                unique.append(d)
        return unique[: self.k]

    @staticmethod
    def _format_context(docs: List[Document]) -> str:
        return "\n\n".join(
            f"Source: {d.metadata.get('source', 'Unknown')}\n{d.page_content}"
            for d in docs
        )

    def invoke(self, inputs: dict) -> dict:
        question = self._standalone_question(inputs)
        docs = self._retrieve(question)
        self.last_context = docs
        context = self._format_context(docs)
        answer = (self.qa_prompt | self.llm | StrOutputParser()).invoke(
            {"input": inputs["input"], "context": context}
        )
        return {"answer": answer, "context": docs}

    def stream(self, inputs: dict) -> Iterator[str]:
        question = self._standalone_question(inputs)
        docs = self._retrieve(question)
        self.last_context = docs
        context = self._format_context(docs)
        answer_chain = self.qa_prompt | self.llm | StrOutputParser()
        for chunk in answer_chain.stream({"input": inputs["input"], "context": context}):
            yield chunk


def create_qa_chain(
    vectorstore,
    provider: str = "groq",
    model_name: str = "llama-3.1-8b-instant",
    api_key: Optional[str] = None,
    k: int = 4,
    doc_sources: Optional[List[str]] = None,
) -> QAChain:
    llm = create_llm(provider, model_name, api_key)

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()
    return QAChain(llm, vectorstore, contextualize_chain, qa_prompt, k, doc_sources)