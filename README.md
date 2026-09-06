# RAG Document Q&A

A chat-with-your-documents application built on **Retrieval-Augmented Generation (RAG)**. Upload PDFs or text files, index them into a local vector database, then ask questions in natural language. The LLM answers strictly from the retrieved document chunks and cites its sources.

Live demo / production target: Vercel (see [Deploying to Vercel](#deploying-to-vercel)).

---

## Features

- **Multi-provider LLM support** – Groq, OpenAI, Anthropic, or local Ollama, switchable from the sidebar without restarting.
- **Persistent online storage (optional)** – with Supabase, vectors live in **pgvector** and uploaded files in **Supabase Storage**, so your index survives restarts/cold starts (important for Vercel). Falls back to local Chroma when Supabase isn't configured.
- **Streaming answers** – token-by-token responses with chat history awareness (follow-up questions work).
- **Source citations** – expand "View retrieved context" to see exactly which document and page each answer used.
- **Document management** – upload files in the UI or drop them into `data/documents/`; re-index anytime.
- **Customizable retrieval** – embedding model, chunk size/overlap, and `k` (number of chunks retrieved) are configurable in the UI.
- **Filter by document** – restrict retrieval to a subset of documents.
- **Similarity retrieval with de-dup** – top-k context with duplicate chunks removed; efficient on both Chroma and pgvector.
- **Chat export** – download the conversation as a `.txt` file.
- **Local & containerized** – runs natively with Python or via Docker (`Dockerfile` + `docker-compose.yml`).

---

## How it works

```
                    ┌─────────────────────────────────────────────┐
                    │                Ingest                       │
                    │  documents/  ──►  load  ──►  chunk  ──►     │
                    │  (PDF/TXT/  Markdown/source code)           │
                    │  Embeddings: fastembed (local, no API)      │
                    │  Store: Chroma (data/chroma_db)             │
                    └─────────────────────────────────────────────┘
                                       │
                    ┌──────────────────▼─────────────────────────┐
                    │                 Query                        │
                    │  user question ─► contextualize (chat hist)  │
                    │       ──► retrieve top-k chunks (MMR)        │
                    │       ──► LLM answers from context only      │
                    │       ──► stream answer + show citations     │
                    └─────────────────────────────────────────────┘
```

1. **Ingest** – documents are loaded (`PyPDFLoader` / `TextLoader`), split into overlapping chunks (`RecursiveCharacterTextSplitter`), embedded locally with [FastEmbed](https://github.com/qdrant/fastembed) (BGE small, no API key needed), and stored in a vector store — **Supabase pgvector** when configured, otherwise local [Chroma](https://www.trychroma.com/).
2. **Query** – the latest question is re-formulated to a standalone question using prior chat history, the top-k most relevant chunks are retrieved by cosine similarity, and only that context is given to the LLM. Retrieved sources are shown under each answer.

---

## Tech stack

| Layer      | Technology                                                                 |
| ---------- | -------------------------------------------------------------------------- |
| UI         | [Streamlit](https://streamlit.io)                                           |
| Orchestration | LangChain (LangChain v1.3+, LCEL chains)                                 |
| Embeddings | FastEmbed (`BAAI/bge-small-en-v1.5`) — runs locally, no API key             |
| Vector DB  | Supabase pgvector (online) or Chroma local (`data/chroma_db/`)              |
| File store | Supabase Storage (optional, online)                                         |
| LLM        | Groq, OpenAI, Anthropic, or Ollama via `langchain-*` providers              |
| Docs       | PDF (`pypdf`) and plain-text/code files                                     |
| Deploy     | Docker, `docker-compose`, Vercel (container image)                          |

## Project structure

```
rag-doc-qa/
├── app/                     # Application package
│   ├── ingest.py            # CLI script: index documents into the active vector store
│   ├── ingestion/
│   │   ├── loader.py        # load documents + chunking
│   │   ├── storage.py       # Supabase Storage helpers (persist uploaded files)
│   │   └── vectorstore.py   # Supabase pgvector / Chroma create-load helpers
│   ├── retrieval/
│   │   └── qa_chain.py      # contextualize → retrieve → LLM (invoke + stream)
│   └── ui/
│       └── streamlit_app.py # Streamlit UI
├── data/
│   ├── documents/           # your source files (PDF/TXT/…) — local fallback
│   └── chroma_db/           # local vector index (auto-generated, git-ignored)
├── supabase/
│   └── schema.sql           # pgvector table + match_documents RPC (run once)
├── Dockerfile               # local/container image (streamlit on :8501)
├── Dockerfile.vercel        # Vercel container image (listens on $PORT)
├── docker-compose.yml       # local compose: mounts ./data and ./.env
├── requirements.txt
├── .env.example             # copy to .env and fill in API keys
└── run.py                   # convenience launcher for streamlit
```

---

## Quickstart (local)

Prerequisites: **Python 3.10+**.

```bash
# 1. Clone and enter the project
git clone https://github.com/EtrossO/rag-doc-qa.git
cd rag-doc-qa

# 2. Create a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env        # then add your API key(s) — see Configuration

# 5. (Optional) pre-index documents from the CLI
python app/ingest.py

# 6. Run the app
streamlit run app/ui/streamlit_app.py
# or:  python run.py
```

Open http://localhost:8501, pick your LLM provider/model in the sidebar, upload documents (or use the ones already in `data/documents/`), click **⚡ Load & Index**, then start asking questions.

> You can also index on demand from the UI without running `app/ingest.py` – the **Load & Index** button does the same thing from the sidebar.

## Configuration

Copy `.env.example` to `.env` and set at least one API key:

| Variable         | Required?           | Description                                                                     |
| ---------------- | ------------------- | ------------------------------------------------------------------------------- |
| `GROQ_API_KEY`   | At least one of the three | [Groq](https://console.groq.com) API key (fast, generous free tier)          |
| `OPENAI_API_KEY` | At least one of the three | [OpenAI](https://platform.openai.com) API key                                 |
| `ANTHROPIC_API_KEY` | At least one of the three | [Anthropic](https://console.anthropic.com) API key                            |
| `EMBEDDING_MODEL` | No                  | FastEmbed model name (default `BAAI/bge-small-en-v1.5`, 384-dim).             |
| `CHUNK_SIZE`      | No                  | Character chunk size (default `1000`).                                         |
| `CHUNK_OVERLAP`   | No                  | Chunk overlap (default `200`).                                                 |
| `MODEL_NAME`      | No                  | Default model hint (the UI offers a provider-specific model list).              |
| `SUPABASE_URL`    | Only for online storage | Supabase project URL (e.g. `https://xxxx.supabase.co`).                     |
| `SUPABASE_SERVICE_ROLE_KEY` | Only for online storage | Supabase service-role key (server-side).                              |
| `SUPABASE_TABLE`  | No                  | Postgres table storing chunks (default `documents`).                           |
| `SUPABASE_BUCKET` | No                  | Storage bucket for uploaded files (default `documents`).                       |

**Embeddings run locally** (FastEmbed downloads `BAAI/bge-small-en-v1.5` on first use) – no embedding API key needed. **Ollama** requires no key at all and runs a local model (e.g. `llama3.2`).

### Persistent storage with Supabase (recommended for online deploys)

Local disk on hosted platforms (Vercel) is ephemeral — the Chroma index and any
in-app uploads disappear when the instance scales down. Supabase fixes that:

1. Create a free project at [supabase.com](https://supabase.com).
2. In **SQL Editor**, run the contents of [`supabase/schema.sql`](supabase/schema.sql)
   (creates the `documents` table, the pgvector extension/index, and the
   `match_documents` retrieval function).
3. Create a Storage bucket named `documents` (Dashboard → Storage → New bucket,
   private). The app can auto-create it too, but manual is deterministic.
4. Add to `.env` (or Vercel env vars): `SUPABASE_URL` and
   `SUPABASE_SERVICE_ROLE_KEY` (Dashboard → Settings → API → Service Role key).
5. Upload documents via the UI — they go into Storage, and **Load & Index**
   reads from Storage, embeds, and stores chunks in pgvector.

Once Supabase is configured, the app **ignores local Chroma entirely**: uploads,
indexing, and retrieval are all online. `python app/ingest.py` also targets
Supabase automatically in that mode.

## Running with Docker

```bash
docker compose up -d --build
```

- Mounts `./data` (documents + chroma index) and `./.env` into the container.
- Serves the app at http://localhost:8501.
- Runs as a non-root user with a health check.

## Deploying to Vercel

Vercel doesn't have a native Streamlit framework, so this repo deploys Streamlit as a **container image** using a `Dockerfile.vercel` (Vercel's supported OCI container path). Vercel builds the image, stores it in the Vercel Container Registry, and serves it from a function that scales automatically.

### One-time setup

1. Push this repo to GitHub (done here).
2. In the [Vercel dashboard](https://vercel.com/new), **Import Project** → select the `EtrossO/rag-doc-qa` repo.
3. In **Project → Settings → Environment Variables** add at least one LLM key, e.g.:
   - `GROQ_API_KEY` (recommended) *or* `OPENAI_API_KEY` *or* `ANTHROPIC_API_KEY`
   - **`SUPABASE_URL`** and **`SUPABASE_SERVICE_ROLE_KEY`** — recommended for persistent indexes/uploads (see [Persistent storage](#persistent-storage-with-supabase-recommended-for-online-deploys)).
   - Optionally `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP` (all have sensible defaults).
4. **Framework Preset** must be **Services** (or import with the repo's `vercel.json` detected) so Vercel builds the Docker image via `Dockerfile.vercel` instead of trying a Python buildpack.
5. Deploy.

### How it works

- Vercel detects `Dockerfile.vercel` at the repo root, builds it, and routes all traffic to the container.
- The container runs Streamlit headless on `$PORT` (Vercel uses port `80` by default; the `CMD` falls back to it).
- Deploys automatically on every push to the connected branch.

### Important caveats for this app on Vercel

- **Storage is ephemeral unless Supabase is configured.** Vercel functions are stateless and scale to zero. With Supabase env vars set, documents persist in Storage and the vector index in pgvector, so a cold start only needs **Load Existing** (no re-upload). Chat history in `st.session_state` is still per-session.
- **WebSocket / duration.** Streamlit uses WebSockets, which Vercel supports on functions in beta; long-running or high-traffic sessions can be cut at a function's max duration and the client rewinds/reconnects. Keep chats short-ish.
- **Embedding model download.** FastEmbed downloads `BAAI/bge-small-en-v1.5` on first index, so the first **Load & Index** takes longer.
- **First run setup.** One-time: run `supabase/schema.sql` in the SQL Editor and create the storage bucket (the app auto-creates the bucket, but manual is deterministic).

**For larger or multi-tenant use**, swap the in-app ingestion for a background indexer (e.g. `python app/ingest.py` in CI or a scheduled job), add auth/RLS on the Supabase tables, and consider a dedicated Streamlit host with persistent disks.

---

## Limitations

- Answers are grounded in the provided documents only; if the answer isn't in context the model says so.
- Without Supabase, the local Chroma store is single-user and machine-bound (no auth, no concurrent-user persistence).
- PDFs are extracted as plain text (`pypdf`) – scanned/image-based PDFs are not OCR'd.

## Roadmap ideas

- Persistent vector storage (Supabase pgvector / hosted Chroma) + auth.
- OCR for scanned PDFs.
- Metadata-aware reranking / hybrid search (keyword + vector).
- Web UI polish and usage metrics.

## License

Private / internal project – no license file yet.