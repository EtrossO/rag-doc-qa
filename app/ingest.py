import os

from dotenv import load_dotenv

from app.ingestion.loader import load_documents, chunk_documents
from app.ingestion.storage import download_documents
from app.ingestion.vectorstore import create_vectorstore, is_supabase_configured

load_dotenv()

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "documents")
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")


def run_ingestion():
    if is_supabase_configured():
        print("Supabase configured: reading documents from online storage...")
        docs = download_documents()
    else:
        docs = []

    if not docs:
        print(f"Loading documents from: {DOCS_DIR}")
        docs = load_documents(DOCS_DIR)

    if not docs:
        raise SystemExit("No documents found to index. Add files to data/documents/ or upload via the app.")

    print(f"Loaded {len(docs)} document(s)")

    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Split into {len(chunks)} chunk(s)")

    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    print("Creating vector store...")
    create_vectorstore(chunks, embedding_model=embedding_model, persist_directory=CHROMA_DIR)
    if is_supabase_configured():
        print("Indexing complete (Supabase pgvector)")
    else:
        print(f"Vector store saved to: {CHROMA_DIR}")
    print("Ingestion complete!")


if __name__ == "__main__":
    run_ingestion()