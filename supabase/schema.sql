-- Supabase schema for rag-doc-qa
-- Run this in: Supabase Dashboard > SQL Editor (or `supabase db push`)
--
-- NOTE: the embedding dimension must match the FastEmbed model used by the app
-- (default BAAI/bge-small-en-v1.5 -> 384 dimensions). If you change
-- EMBEDDING_MODEL, recreate the table with the matching dimension.

-- 1. Enable pgvector
create extension if not exists vector;

-- 2. Chunks table: one row per document chunk
create table if not exists documents (
    id         bigserial primary key,
    content    text,
    metadata   jsonb       default '{}'::jsonb,
    embedding  vector(384)
);

-- 3. HNSW index for fast cosine similarity search (works on empty tables,
--    so it is safe to run before ingesting any documents).
create index if not exists documents_embedding_idx
    on documents using hnsw (embedding vector_cosine_ops);

-- 4. Retrieval function consumed by LangChain's SupabaseVectorStore.
--    Filtering is exact-match on metadata keys (e.g. {"source": "file.pdf"}).
create or replace function match_documents(
    query_embedding vector(384),
    match_count     int,
    filter          jsonb default '{}'
)
returns table (
    id         bigint,
    content    text,
    metadata   jsonb,
    similarity float
)
language plpgsql
as $$
begin
    return query
    select
        documents.id,
        documents.content,
        documents.metadata,
        1 - (documents.embedding <=> query_embedding) as similarity
    from documents
    where documents.metadata @> filter
    order by documents.embedding <=> query_embedding
    limit match_count;
end;
$$;