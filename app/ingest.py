import os
import sys

from dotenv import load_dotenv

from app.ingestion.loader import load_documents, chunk_documents
from app.ingestion.vectorstore import create_vectorstore

load_dotenv()

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "documents")
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")


def run_ingestion():
    print(f"Loading documents from: {DOCS_DIR}")
    docs = load_documents(DOCS_DIR)
    print(f"Loaded {len(docs)} document(s)")

    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Split into {len(chunks)} chunk(s)")

    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    print("Creating vector store...")
    create_vectorstore(chunks, CHROMA_DIR, embedding_model=embedding_model)
    print(f"Vector store saved to: {CHROMA_DIR}")
    print("Ingestion complete!")


if __name__ == "__main__":
    run_ingestion()
