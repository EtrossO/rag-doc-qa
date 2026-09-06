import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from app.ingestion.loader import load_documents, chunk_documents
from app.ingestion.storage import (
    download_documents,
    ensure_bucket,
    list_document_names,
    storage_ready,
    upload_document,
)
from app.ingestion.vectorstore import (
    count_documents,
    create_vectorstore,
    is_supabase_configured,
    load_vectorstore,
)
from app.retrieval.qa_chain import create_qa_chain

load_dotenv()

if storage_ready():
    ensure_bucket()

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = PROJECT_DIR / "data" / "documents"
CHROMA_DIR = PROJECT_DIR / "data" / "chroma_db"

DOCS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="RAG Document Q&A", page_icon="", layout="wide")
st.markdown("""
<style>
.stChatMessage p, .stChatMessage li, .stChatMessage div {
    text-align: justify;
}
</style>
""", unsafe_allow_html=True)
st.title(" Document Q&A with RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False
if "last_provider" not in st.session_state:
    st.session_state.last_provider = None


def get_available_docs() -> list:
    if is_supabase_configured():
        return list_document_names()
    if not DOCS_DIR.exists():
        return []
    return sorted([f.name for f in DOCS_DIR.iterdir() if f.is_file()])


def reset_chain():
    st.session_state.chain = None
    st.session_state.vectorstore_ready = False


def clear_chat():
    st.session_state.messages = []


def build_chat_export() -> str:
    lines = []
    lines.append("RAG Document Q&A - Chat Export")
    lines.append(f"Exported: {datetime.now().isoformat()}")
    lines.append("=" * 60)
    for msg in st.session_state.messages:
        role = msg["role"].upper()
        content = msg["content"]
        lines.append(f"\n[{role}]")
        lines.append(content)
    return "\n".join(lines)


with st.sidebar:
    if is_supabase_configured():
        st.info(" Online storage active: vectors in Supabase pgvector, files in Supabase Storage", icon="")
    else:
        st.warning("Local-only mode: index and uploads are ephemeral on hosted platforms. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY for persistence.", icon="")

    st.header(" Provider Configuration")

    provider = st.selectbox(
        "LLM Provider",
        options=["groq", "openai", "anthropic", "ollama"],
        index=0,
        on_change=reset_chain,
    )

    provider_models = {
        "groq": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
        "ollama": ["llama3.2", "mistral", "phi3", "qwen2.5"],
    }

    model_name = st.selectbox(
        "Model",
        options=provider_models.get(provider, []),
        index=0,
    )

    api_key = os.getenv(f"{provider.upper()}_API_KEY") if provider != "ollama" else None
    if provider != "ollama" and not api_key:
        st.warning(f"Set {provider.upper()}_API_KEY in .env", icon="")

    st.divider()

    st.header(" Document Management")

    uploaded_files = st.file_uploader(
        "Upload PDF or text files",
        type=["pdf", "txt", "md", "py", "js", "ts", "html", "css", "json", "yaml", "yml"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        saved = 0
        uploaded = 0
        for f in uploaded_files:
            content = f.getbuffer()
            dest = DOCS_DIR / f.name
            if not dest.exists():
                dest.write_bytes(content)
                saved += 1
            if storage_ready() and upload_document(bytes(content), f.name):
                uploaded += 1
        if saved or uploaded:
            st.success(f"Saved {saved} file(s) locally, uploaded {uploaded} to online storage.")
            reset_chain()
        else:
            st.info("All files already exist in documents folder.")

    embedding_model = st.text_input(
        "Embedding Model",
        value=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
    )

    chunk_size = st.number_input("Chunk Size", min_value=200, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, value=200, step=50)
    k_retrieval = st.number_input("Documents to retrieve (k)", min_value=1, max_value=20, value=4, step=1)

    available_docs = get_available_docs()
    doc_sources = None
    if available_docs:
        st.divider()
        st.header(" Filter by Document")
        selected_docs = st.multiselect(
            "Select documents to search (empty = search all)",
            options=available_docs,
        )
        if selected_docs:
            doc_sources = list(selected_docs)

    col1, col2 = st.columns(2)
    with col1:
        if st.button(" Load & Index", use_container_width=True, type="primary"):
            with st.spinner("Loading documents..."):
                try:
                    docs = download_documents() if is_supabase_configured() else []
                    if not docs:
                        docs = load_documents(str(DOCS_DIR))
                    if not docs:
                        st.error(f"No documents found in {DOCS_DIR}. Upload files first.")
                    else:
                        chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                        vs = create_vectorstore(
                            chunks,
                            embedding_model=embedding_model,
                            persist_directory=str(CHROMA_DIR),
                        )
                        chain = create_qa_chain(
                            vs,
                            provider=provider,
                            model_name=model_name,
                            api_key=api_key,
                            k=k_retrieval,
                            doc_sources=doc_sources,
                        )
                        st.session_state.chain = chain
                        st.session_state.vectorstore_ready = True
                        st.success(f"Indexed {len(docs)} doc(s) → {len(chunks)} chunk(s)")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to index: {e}")

    with col2:
        if st.button(" Load Existing", use_container_width=True):
            with st.spinner("Loading vector store..."):
                try:
                    total = count_documents(persist_directory=str(CHROMA_DIR))
                    if total == 0:
                        st.error("No existing index found. Ingest documents first.")
                    else:
                        vs = load_vectorstore(
                            embedding_model=embedding_model,
                            persist_directory=str(CHROMA_DIR),
                        )
                        chain = create_qa_chain(
                            vs,
                            provider=provider,
                            model_name=model_name,
                            api_key=api_key,
                            k=k_retrieval,
                            doc_sources=doc_sources,
                        )
                        st.session_state.chain = chain
                        st.session_state.vectorstore_ready = True
                        st.success(f"Loaded index with {total} chunks")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to load index: {e}")

    st.divider()
    st.markdown(f"**Documents folder:** `{DOCS_DIR}`")
    st.markdown(f"Files found: **{len(available_docs)}**")

    if st.button(" Clear Chat", use_container_width=True):
        clear_chat()
        st.rerun()

    if st.session_state.messages:
        chat_export = build_chat_export()
        st.download_button(
            " Export Chat",
            data=chat_export,
            file_name=f"rag-chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

st.subheader(" Ask questions about your documents")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.vectorstore_ready:
        response = "Please load or index documents first using the sidebar."
        with st.chat_message("assistant"):
            st.markdown(response)
    else:
        with st.chat_message("assistant"):
            response = ""
            context = []
            try:
                chain = st.session_state.chain
                response = st.write_stream(chain.stream({
                    "input": prompt,
                    "chat_history": [
                        (m["role"], m["content"]) for m in st.session_state.messages[:-1]
                    ],
                }))
                context = chain.last_context
            except Exception as e:
                response = f"Error generating response: {e}"
                st.markdown(response)

            if context:
                with st.expander("View retrieved context"):
                    for i, doc in enumerate(context):
                        source = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", "")

                        label = f"**Source {i+1}:** `{Path(source).name}`"
                        if page != "":
                            label += f" (page {int(page) + 1})"
                        st.markdown(label)

                        preview = doc.page_content[:600]
                        if len(doc.page_content) > 600:
                            preview += "..."
                        st.markdown(f"> {preview}")
                        st.divider()

    st.session_state.messages.append({"role": "assistant", "content": response})
