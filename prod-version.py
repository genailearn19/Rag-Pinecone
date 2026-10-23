
"""
Production-oriented asynchronous ingestion pipeline:
Documents -> Chunks -> OpenAI Embeddings -> Pinecone

Features:
- Async OpenAI embedding requests with bounded concurrency
- Deterministic document and chunk IDs
- Character-based chunking with overlap
- Batched embeddings and controlled Pinecone upserts
- Retry with exponential backoff and jitter
- Metadata, namespaces, and metadata filtering
- Logging, failure tracking, token usage, timings, and index stats
"""

import asyncio
import hashlib
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any
from pathlib import Path
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ============================================================
# Configuration
# ============================================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

INDEX_NAME = "prod-rag-pinecone-index"
NAMESPACE = "knowledge-base"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
METRIC = "cosine"

CLOUD = "aws"
REGION = "us-east-1"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

EMBEDDING_BATCH_SIZE = 64
UPSERT_BATCH_SIZE = 100
MAX_CONCURRENT_EMBEDDING_REQUESTS = 3

MAX_RETRIES = 5
RETRY_BASE_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 60

# Optional: set your actual billing rate in .env.
# Leave unset to disable cost estimation.
EMBEDDING_PRICE_PER_1M_TOKENS = os.getenv(
    "EMBEDDING_PRICE_PER_1M_TOKENS"
)

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise RuntimeError(
        "Set PINECONE_API_KEY and OPENAI_API_KEY in your .env file."
    )

if CHUNK_SIZE <= 0 or not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError("Invalid chunk size or overlap configuration.")

if EMBEDDING_BATCH_SIZE <= 0 or UPSERT_BATCH_SIZE <= 0:
    raise ValueError("Batch sizes must be positive.")

if MAX_CONCURRENT_EMBEDDING_REQUESTS <= 0:
    raise ValueError("Embedding concurrency must be positive.")

# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("pinecone-ingestion")

# ============================================================
# Clients
# ============================================================

pc = Pinecone(api_key=PINECONE_API_KEY)

openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    timeout=REQUEST_TIMEOUT_SECONDS,
    max_retries=0,  # Our retry policy is implemented below.
)

embedding_semaphore = asyncio.Semaphore(
    MAX_CONCURRENT_EMBEDDING_REQUESTS
)

PDF_PATH = Path(
    r"E:\GEN-AI-ESWARA\Rag-Pinecone\documents\Spring-Core.pdf"
)

def load_pdf_document(pdf_path: Path) -> dict:
    """Extract text from a PDF and prepare it for ingestion."""

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if not pdf_path.is_file():
        raise ValueError(
            f"The supplied path is not a file: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("The input file must be a PDF.")

    reader = PdfReader(str(pdf_path))

    if reader.is_encrypted:
        raise ValueError(
            "The PDF is encrypted. Provide an accessible PDF."
        )

    extracted_pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""

            if page_text.strip():
                extracted_pages.append(
                    f"[Page {page_number}]\n{page_text}"
                )

        except Exception as exc:
            raise RuntimeError(
                f"Failed to extract PDF page {page_number}"
            ) from exc

    full_text = "\n\n".join(extracted_pages).strip()

    if not full_text:
        raise ValueError(
            "No extractable text found. "
            "This may be a scanned PDF requiring OCR."
        )

    return {
        "document_id": pdf_path.stem,
        "source": pdf_path.name,
        "category": "pdf-document",
        "text": full_text,
    }


# Compatible with the existing ingest_documents() function
DOCUMENTS = [load_pdf_document(PDF_PATH)]

print("PDF loaded successfully")
print("Document ID:", DOCUMENTS[0]["document_id"])
print("Source:", DOCUMENTS[0]["source"])
print("Extracted characters:", len(DOCUMENTS[0]["text"]))
# ============================================================
# Data model
# ============================================================

@dataclass
class TextChunk:
    vector_id: str
    document_id: str
    chunk_index: int
    text: str
    source: str
    category: str


# ============================================================
# Stable IDs
# ============================================================

def stable_document_id(document: dict[str, Any]) -> str:
    """
    Prefer a permanent source-system ID.

    If the source system does not provide one, use a content hash.
    Content-hash IDs change when the document content changes.
    """
    supplied_id = document.get("document_id")

    if supplied_id:
        return str(supplied_id)

    return hashlib.sha256(
        document["text"].encode("utf-8")
    ).hexdigest()


def stable_chunk_id(document_id: str, chunk_index: int) -> str:
    """
    The same document and chunk index produce the same vector ID.
    Re-ingesting unchanged chunks overwrites them instead of
    creating duplicate vectors.
    """
    raw_id = f"{document_id}:chunk:{chunk_index}"

    return hashlib.sha256(
        raw_id.encode("utf-8")
    ).hexdigest()


# ============================================================
# Chunking
# ============================================================

def split_into_chunks(text: str) -> list[str]:
    """
    Character-based chunking with overlap.

    CHUNK_SIZE is a character limit, not a token limit.
    For production document parsing, consider a tokenizer-aware
    splitter to control embedding token usage more precisely.
    """
    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def prepare_chunks(
    documents: list[dict[str, Any]],
) -> list[TextChunk]:
    """Validate source documents and create stable chunk records."""
    result = []

    for document in documents:
        text = document.get("text", "").strip()

        if not text:
            logger.warning(
                "Skipping empty document: %s",
                document.get("document_id", "unknown"),
            )
            continue

        document_id = stable_document_id(document)
        source = str(document.get("source", "unknown"))
        category = str(document.get("category", "general"))

        for chunk_index, chunk_text in enumerate(
            split_into_chunks(text)
        ):
            vector_id = stable_chunk_id(
                document_id, chunk_index
            )

            result.append(
                TextChunk(
                    vector_id=vector_id,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    source=source,
                    category=category,
                )
            )

    return result


# ============================================================
# Pinecone index initialization
# ============================================================

def initialize_index():
    """Create the index if absent and verify its configuration."""
    if not pc.has_index(INDEX_NAME):
        logger.info("Creating Pinecone index: %s", INDEX_NAME)

        pc.create_index(
            name=INDEX_NAME,
            vector_type="dense",
            dimension=EMBEDDING_DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(
                cloud=CLOUD,
                region=REGION,
            ),
        )

    # Wait until the newly created index is ready.
    description = pc.describe_index(INDEX_NAME)

    deadline = time.monotonic() + 120

    while not description.status["ready"]:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Pinecone index {INDEX_NAME} was not ready in time."
            )

        time.sleep(2)
        description = pc.describe_index(INDEX_NAME)

    if description.dimension != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Index dimension is {description.dimension}; "
            f"expected {EMBEDDING_DIMENSION}."
        )

    logger.info("Pinecone index is ready: %s", INDEX_NAME)

    return pc.Index(INDEX_NAME)


# ============================================================
# Retry helpers
# ============================================================

def is_retryable(error: Exception) -> bool:
    """Retry temporary network, timeout, throttling, and server errors."""
    if isinstance(
        error,
        (APIConnectionError, APITimeoutError, RateLimitError),
    ):
        return True

    if isinstance(error, APIStatusError):
        return (
            error.status_code in (408, 409, 429)
            or error.status_code >= 500
        )

    return False


async def async_with_retry(operation, description: str):
    """Retry asynchronous operations with exponential backoff and jitter."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await operation()

        except Exception as error:
            if not is_retryable(error) or attempt >= MAX_RETRIES:
                raise

            delay = RETRY_BASE_SECONDS * (2 ** attempt)
            delay = min(delay, 30.0) + random.uniform(0, 0.5)

            logger.warning(
                "%s failed (attempt %s/%s): %s. "
                "Retrying in %.2f seconds.",
                description,
                attempt + 1,
                MAX_RETRIES + 1,
                error,
                delay,
            )

            await asyncio.sleep(delay)

    raise RuntimeError(f"Retry loop exhausted: {description}")


def sync_with_retry(operation, description: str):
    """Retry synchronous Pinecone operations."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return operation()

        except Exception as error:
            if not is_retryable(error) or attempt >= MAX_RETRIES:
                raise

            delay = min(
                RETRY_BASE_SECONDS * (2 ** attempt), 30.0
            ) + random.uniform(0, 0.5)

            logger.warning(
                "%s failed (attempt %s/%s): %s. "
                "Retrying in %.2f seconds.",
                description,
                attempt + 1,
                MAX_RETRIES + 1,
                error,
                delay,
            )

            time.sleep(delay)

    raise RuntimeError(f"Retry loop exhausted: {description}")


# ============================================================
# Batched embedding generation
# ============================================================

async def generate_embeddings(
    texts: list[str],
) -> tuple[list[list[float]], int | None]:
    """Generate embeddings for one bounded batch."""
    async def request():
        async with embedding_semaphore:
            return await openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
                dimensions=EMBEDDING_DIMENSION,
            )

    response = await async_with_retry(
        request, "OpenAI embedding request"
    )

    # API response data is ordered by its input index.
    ordered_data = sorted(
        response.data, key=lambda item: item.index
    )

    embeddings = [item.embedding for item in ordered_data]

    if len(embeddings) != len(texts):
        raise RuntimeError(
            "Embedding count does not match input text count."
        )

    for embedding in embeddings:
        if len(embedding) != EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Unexpected embedding dimension: {len(embedding)}"
            )

    usage = getattr(response, "usage", None)
    token_count = (
        getattr(usage, "total_tokens", None)
        if usage is not None
        else None
    )

    return embeddings, token_count


# ============================================================
# Pinecone upsert
# ============================================================

def upsert_vectors(index, vectors: list[dict[str, Any]]):
    """Upload one bounded vector batch, with retries."""
    return sync_with_retry(
        lambda: index.upsert(
            vectors=vectors,
            namespace=NAMESPACE,
        ),
        "Pinecone upsert",
    )


# ============================================================
# Asynchronous ingestion pipeline
# ============================================================

async def ingest_documents(
    index,
    documents: list[dict[str, Any]],
):
    started_at = time.perf_counter()

    chunks = prepare_chunks(documents)

    if not chunks:
        logger.warning("No valid document chunks to ingest.")
        return

    logger.info(
        "Prepared %s chunks from %s documents.",
        len(chunks),
        len(documents),
    )

    stats = {
        "chunks_total": len(chunks),
        "chunks_uploaded": 0,
        "chunks_failed": 0,
        "embedding_batches_succeeded": 0,
        "upsert_batches_succeeded": 0,
        "embedding_tokens": 0,
        "embedding_token_usage_known": True,
    }

    # Divide input into bounded embedding batches.
    embedding_batches = [
        chunks[i:i + EMBEDDING_BATCH_SIZE]
        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE)
    ]

    async def process_batch(batch_number, batch):
        try:
            texts = [chunk.text for chunk in batch]

            embeddings, tokens = await generate_embeddings(texts)

            stats["embedding_batches_succeeded"] += 1

            if tokens is None:
                stats["embedding_token_usage_known"] = False
            else:
                stats["embedding_tokens"] += tokens

            vectors = []

            for chunk, embedding in zip(batch, embeddings):
                vectors.append({
                    "id": chunk.vector_id,
                    "values": embedding,
                    "metadata": {
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "source": chunk.source,
                        "category": chunk.category,
                        "text": chunk.text,
                    },
                })

            # Upsert in smaller, controlled batches.
            for start in range(0, len(vectors), UPSERT_BATCH_SIZE):
                vector_batch = vectors[
                    start:start + UPSERT_BATCH_SIZE
                ]

                # Pinecone SDK calls are synchronous, so move them
                # off the asyncio event loop.
                await asyncio.to_thread(
                    upsert_vectors, index, vector_batch
                )

                stats["upsert_batches_succeeded"] += 1
                stats["chunks_uploaded"] += len(vector_batch)

            logger.info(
                "Batch %s/%s uploaded: %s chunks",
                batch_number,
                len(embedding_batches),
                len(batch),
            )

        except Exception:
            stats["chunks_failed"] += len(batch)

            logger.exception(
                "Batch %s failed. Chunk IDs: %s",
                batch_number,
                [chunk.vector_id for chunk in batch],
            )

    # The semaphore bounds simultaneous embedding requests.
    # gather lets independent batches progress concurrently.
    await asyncio.gather(
        *[
            process_batch(batch_number, batch)
            for batch_number, batch in enumerate(
                embedding_batches, start=1
            )
        ]
    )

    elapsed = time.perf_counter() - started_at

    stats["elapsed_seconds"] = round(elapsed, 2)
    stats["chunks_per_second"] = round(
        stats["chunks_uploaded"] / elapsed, 2
    ) if elapsed else 0

    logger.info("Ingestion summary: %s", stats)

    if (
        EMBEDDING_PRICE_PER_1M_TOKENS
        and stats["embedding_token_usage_known"]
    ):
        estimated_cost = (
            stats["embedding_tokens"]
            / 1_000_000
            * float(EMBEDDING_PRICE_PER_1M_TOKENS)
        )

        logger.info(
            "Estimated embedding cost: %.6f "
            "(same currency as configured rate)",
            estimated_cost,
        )

    # Report failures instead of silently declaring success.
    if stats["chunks_failed"] > 0:
        raise RuntimeError(
            f"Ingestion incomplete: {stats['chunks_failed']} "
            "chunks failed. Review logs and rerun failed work."
        )

    return stats


# ============================================================
# Retrieval example: metadata filtering
# ============================================================

async def search_vectors(
    index,
    question: str,
    category: str | None = None,
    top_k: int = 5,
):
    """Search a namespace, optionally filtering by category."""
    embeddings, _ = await generate_embeddings([question])

    query_args = {
        "namespace": NAMESPACE,
        "vector": embeddings[0],
        "top_k": top_k,
        "include_metadata": True,
    }

    if category:
        query_args["filter"] = {
            "category": {"$eq": category}
        }

    response = await asyncio.to_thread(
        lambda: index.query(**query_args)
    )

    for match in response.matches:
        metadata = match.metadata or {}

        logger.info(
            "Search result | score=%s | source=%s | text=%s",
            match.score,
            metadata.get("source"),
            metadata.get("text"),
        )

    return response


# ============================================================
# OpenAI LLM Augmentation
# ============================================================

LLM_MODEL = "gpt-4o-mini"


async def generate_llm_answer(question: str, search_results) -> str:
    """
    Generate an answer using the user's question and
    the top matching chunks retrieved from Pinecone.
    """

    context_parts = []

    # Extract text from Pinecone's top matching chunks
    for i, match in enumerate(search_results.matches, start=1):
        metadata = match.metadata or {}
        chunk_text = metadata.get("text", "")

        if chunk_text:
            source = metadata.get("source", "Unknown PDF")

            context_parts.append(
                f"[Source {i}: {source}]\n{chunk_text}"
            )

    context = "\n\n".join(context_parts)

    # Avoid calling the LLM if retrieval returned no content
    if not context.strip():
        return (
            "No relevant PDF content was retrieved from Pinecone. "
            "Please check your indexed documents and search filters."
        )

    # Send the question and retrieved context to OpenAI
    response = await openai_client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a technical documentation assistant. "
                    "Answer using the retrieved PDF context. "
                    "Do not invent facts or claim that information "
                    "exists in the documents when it does not. "
                    "If the context is insufficient, clearly say so. "
                    "Mention source numbers where appropriate."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Retrieved PDF context:\n{context}\n\n"
                    "Provide a clear, structured answer."
                ),
            },
        ],
    )

    return (
        response.choices[0].message.content
        or "OpenAI returned an empty response."
    )


# ============================================================
# Entry point
# ============================================================

async def main():
    index = await asyncio.to_thread(initialize_index)

    try:
        # Step 1: Ingest PDF documents into Pinecone
        # await ingest_documents(index, DOCUMENTS)

        # Step 2: Define the user's question
        question = "Spring transaction management best practices"

        # Step 3: Retrieve the top 5 matching PDF chunks
        search_results = await search_vectors(
            index=index,
            question=question,
            category="pdf-document",
            top_k=5,
        )

        # Step 4: Augment the retrieved context using OpenAI LLM
        answer = await generate_llm_answer(
            question=question,
            search_results=search_results,
        )

        # Step 5: Display the final LLM-generated answer
        print("\n" + "=" * 70)
        print("USER QUESTION")
        print("=" * 70)
        print(question)

        print("\n" + "=" * 70)
        print("OPENAI LLM ANSWER")
        print("=" * 70)
        print(answer)

        # Step 6: Display Pinecone index statistics
        stats = await asyncio.to_thread(
            index.describe_index_stats
        )

        logger.info("Pinecone index statistics: %s", stats)

    finally:
        # Step 7: Close the OpenAI client
        await openai_client.close()


if __name__ == "__main__":
    asyncio.run(main())