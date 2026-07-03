from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma


def create_vectorstore(
    documents: List[Document],
    persist_directory: str,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
) -> Chroma:
    embeddings = FastEmbedEmbeddings(model_name=embedding_model)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    return vectorstore


def load_vectorstore(
    persist_directory: str,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
) -> Optional[Chroma]:
    embeddings = FastEmbedEmbeddings(model_name=embedding_model)
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    return vectorstore
