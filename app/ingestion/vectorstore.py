import os
from pathlib import Path
from typing import List, Optional

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document

TABLE_NAME = os.getenv("SUPABASE_TABLE", "documents")


def is_supabase_configured() -> bool:
    """True when the app should persist to Supabase instead of local Chroma."""
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"))


def get_supabase_client():
    from supabase import create_client

    if not is_supabase_configured():
        raise ValueError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are not set")
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))


def _with_basename_source(documents: List[Document]) -> List[Document]:
    """Store ``source`` as the bare file name (portable across machines/cloud)."""
    for doc in documents:
        src = doc.metadata.get("source")
        if src:
            doc.metadata["source"] = Path(str(src)).name
    return documents


def create_vectorstore(
    documents: List[Document],
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    persist_directory: Optional[str] = None,
    replace_existing: bool = True,
):
    """Index documents into Supabase pgvector (or local Chroma as a fallback).

    Returns a vector store object exposing ``similarity_search`` and
    ``as_retriever``-compatible behavior.
    """
    documents = _with_basename_source(list(documents))
    embeddings = FastEmbedEmbeddings(model_name=embedding_model)

    if is_supabase_configured():
        client = get_supabase_client()
        vectorstore = SupabaseVectorStore(
            client=client,
            embedding=embeddings,
            table_name=TABLE_NAME,
        )
        if replace_existing:
            sources = sorted(
                {d.metadata.get("source") for d in documents if d.metadata.get("source")}
            )
            for source in sources:
                client.table(TABLE_NAME).delete().eq("metadata->>source", source).execute()
        vectorstore.add_documents(documents)
        return vectorstore

    from langchain_chroma import Chroma

    if persist_directory is None:
        raise ValueError("persist_directory is required when Supabase is not configured")
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
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    persist_directory: Optional[str] = None,
):
    """Connect to an existing index without (re)ingesting anything."""
    embeddings = FastEmbedEmbeddings(model_name=embedding_model)

    if is_supabase_configured():
        return SupabaseVectorStore(
            client=get_supabase_client(),
            embedding=embeddings,
            table_name=TABLE_NAME,
        )

    from langchain_chroma import Chroma

    if persist_directory is None:
        raise ValueError("persist_directory is required when Supabase is not configured")
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )


def count_documents(persist_directory: Optional[str] = None) -> int:
    """Number of stored chunks in the active backend."""
    if is_supabase_configured():
        try:
            res = get_supabase_client().table(TABLE_NAME).select("id", count="exact").limit(0).execute()
            return int(res.count or 0)
        except Exception as e:
            print(f"Warning: could not count documents: {e}")
            return 0

    vs = load_vectorstore(persist_directory=persist_directory)
    try:
        return len(vs.get()["ids"])
    except Exception:
        return 0