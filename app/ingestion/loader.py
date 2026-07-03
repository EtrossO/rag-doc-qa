import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(docs_dir: str) -> List[Document]:
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Directory not found: {docs_dir}")

    documents = []
    for file_path in docs_dir.glob("*"):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
                documents.extend(loader.load())
            elif ext in (".txt", ".md", ".py", ".js", ".ts", ".html", ".css", ".json", ".yaml", ".yml"):
                loader = TextLoader(str(file_path), encoding="utf-8")
                documents.extend(loader.load())
            else:
                print(f"Skipping unsupported file: {file_path.name}")
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")

    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)
