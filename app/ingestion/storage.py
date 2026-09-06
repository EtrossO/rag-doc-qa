import os
import tempfile
from pathlib import Path
from typing import List

from app.ingestion.loader import load_documents
from app.ingestion.vectorstore import get_supabase_client

BUCKET = os.getenv("SUPABASE_BUCKET", "documents")


def storage_ready() -> bool:
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"))


def _storage():
    return get_supabase_client().storage.from_(BUCKET)


def ensure_bucket() -> bool:
    """Create the storage bucket if it does not exist yet."""
    if not storage_ready():
        return False
    try:
        client = get_supabase_client()
        try:
            client.storage.get_bucket(BUCKET)
        except Exception:
            client.storage.create_bucket(BUCKET, {"public": False})
        return True
    except Exception as e:
        print(f"Warning: could not ensure storage bucket '{BUCKET}': {e}")
        return False


def list_document_names() -> List[str]:
    """File names persisted in Supabase Storage (source \"documents\" bucket)."""
    if not storage_ready():
        return []
    try:
        items = _storage().list(path="")
        return sorted(i["name"] for i in items if i.get("name") and i.get("id"))
    except Exception as e:
        print(f"Warning: could not list storage files: {e}")
        return []


def upload_document(data: bytes, filename: str) -> bool:
    if not storage_ready():
        return False
    try:
        _storage().upload(filename, data)
        return True
    except Exception as e:
        print(f"Warning: upload failed for {filename}: {e}")
        return False


def download_documents() -> List:
    """Download all persisted documents and load/parse them.

    Returns a list of LangChain Documents with ``source`` as the bare file name.
    """
    if not storage_ready():
        return []
    names = list_document_names()
    if not names:
        return []

    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            try:
                res = _storage().download(name)
                data = res.content if hasattr(res, "content") else res
                (Path(tmp) / name).write_bytes(data)
            except Exception as e:
                print(f"Warning: skipped {name}: {e}")

        docs = load_documents(tmp)

    for doc in docs:
        src = doc.metadata.get("source")
        if src:
            doc.metadata["source"] = Path(str(src)).name
    return docs