from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
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


def create_qa_chain(vectorstore: Chroma, model_name: str = "llama-3.1-8b-instant"):
    llm = ChatGroq(model=model_name, temperature=0)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    question_answer_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, question_answer_prompt)

    history_aware_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, history_aware_prompt
    )

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever, document_chain
    )

    return retrieval_chain
