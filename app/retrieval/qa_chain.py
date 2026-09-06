from typing import Iterator, List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_chroma import Chroma


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
    call are exposed on ``last_context`` so the UI can render citations."""

    def __init__(self, llm, retriever, contextualize_chain, qa_prompt, k):
        self.llm = llm
        self.retriever = retriever
        self.contextualize_chain = contextualize_chain
        self.qa_prompt = qa_prompt
        self.k = k
        self.last_context: List[Document] = []

    def _standalone_question(self, inputs: dict) -> str:
        if inputs.get("chat_history"):
            return self.contextualize_chain.invoke(inputs)
        return inputs["input"]

    def _retrieve(self, question: str) -> List[Document]:
        docs = self.retriever.invoke(question)
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
    vectorstore: Chroma,
    provider: str = "groq",
    model_name: str = "llama-3.1-8b-instant",
    api_key: Optional[str] = None,
    k: int = 4,
    doc_filter: Optional[dict] = None,
) -> QAChain:
    llm = create_llm(provider, model_name, api_key)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 3, "filter": doc_filter},
    )

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
    return QAChain(llm, retriever, contextualize_chain, qa_prompt, k)
