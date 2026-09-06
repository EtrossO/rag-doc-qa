from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma


def create_vectorstore(
    documents: List[Document],
    persist_directory: str,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    replace_existing: bool = True,
) -> Chroma:
    embeddings = FastEmbedEmbeddings(model_name=embedding_model)
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    # Re-indexing should replace, not duplicate. Drop any previously stored
    # chunks whose source matches a document we are about to (re)add so that
    # clicking "Load & Index" again does not keep appending copies.
    if replace_existing:
        sources = sorted(
            {d.metadata.get("source") for d in documents if d.metadata.get("source")}
        )
        if sources:
            try:
                vectorstore._collection.delete(where={"source": {"$in": sources}})
            except Exception as e:
                print(f"Warning: could not clear previous chunks for sources: {e}")

    vectorstore.add_documents(documents)
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
