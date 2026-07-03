from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
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

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    def run(inputs):
        chat_history = inputs.get("chat_history", [])

        if chat_history:
            standalone_question = (
                contextualize_q_prompt | llm | StrOutputParser()
            ).invoke(inputs)
        else:
            standalone_question = inputs["input"]

        docs = retriever.invoke(standalone_question)

        context = "\n\n".join(
            f"Source: {d.metadata.get('source', 'Unknown')}\n{d.page_content}"
            for d in docs
        )

        answer = (qa_prompt | llm | StrOutputParser()).invoke({
            "input": inputs["input"],
            "context": context,
        })

        return {
            "answer": answer,
            "context": docs,
        }

    return RunnableLambda(run)
