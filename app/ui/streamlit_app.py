import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from app.ingestion.loader import load_documents, chunk_documents
from app.ingestion.vectorstore import create_vectorstore, load_vectorstore
from app.retrieval.qa_chain import create_qa_chain

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = PROJECT_DIR / "data" / "documents"
CHROMA_DIR = PROJECT_DIR / "data" / "chroma_db"

st.set_page_config(page_title="📄 RAG Document Q&A", page_icon="📄", layout="wide")
st.title("📄 Document Q&A with RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

with st.sidebar:
    st.header("📁 Document Management")

    embedding_model = st.text_input(
        "Embedding Model",
        value=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
    )
    model_name = st.text_input(
        "LLM Model",
        value=os.getenv("MODEL_NAME", "llama-3.1-8b-instant"),
    )
    chunk_size = st.number_input("Chunk Size", min_value=200, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, value=200, step=50)

    if st.button("🔄 Load & Index Documents", use_container_width=True):
        with st.spinner("Loading documents..."):
            docs = load_documents(str(DOCS_DIR))
            if not docs:
                st.error(f"No documents found in {DOCS_DIR}. Add PDF or text files.")
            else:
                chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                vs = create_vectorstore(chunks, str(CHROMA_DIR), embedding_model=embedding_model)
                chain = create_qa_chain(vs, model_name=model_name)
                st.session_state.chain = chain
                st.session_state.vectorstore_ready = True
                st.success(f"✅ Indexed {len(docs)} doc(s) → {len(chunks)} chunk(s)")

    if st.button("📂 Load Existing Index", use_container_width=True):
        with st.spinner("Loading vector store..."):
            vs = load_vectorstore(str(CHROMA_DIR), embedding_model=embedding_model)
            if vs is None or vs._collection.count() == 0:
                st.error("No existing index found. Ingest documents first.")
            else:
                chain = create_qa_chain(vs, model_name=model_name)
                st.session_state.chain = chain
                st.session_state.vectorstore_ready = True
                st.success(f"✅ Loaded index with {vs._collection.count()} chunks")

    st.divider()
    st.markdown("**📂 Documents folder:**")
    st.code(str(DOCS_DIR))

    doc_count = len(list(DOCS_DIR.glob("*"))) if DOCS_DIR.exists() else 0
    st.markdown(f"Files found: **{doc_count}**")

st.subheader("💬 Ask questions about your documents")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.vectorstore_ready:
        response = "⚠️ Please load documents first using the sidebar."
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chain.invoke({
                    "input": prompt,
                    "chat_history": [
                        (m["role"], m["content"]) for m in st.session_state.messages[:-1]
                    ],
                })
                response = result["answer"]
                st.markdown(response)

                with st.expander("🔍 View retrieved context"):
                    for i, doc in enumerate(result["context"]):
                        source = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", "")
                        label = f"**Source {i+1}:** `{Path(source).name}`"
                        if page:
                            label += f" (page {page})"
                        st.markdown(label)
                        st.markdown(f"> {doc.page_content[:500]}...")
                        st.divider()

    st.session_state.messages.append({"role": "assistant", "content": response})
