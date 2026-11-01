
import os
import re
import json
import logging
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is required.")

GEMINI_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768

FAQ_FILE = os.getenv("FAQ_FILE", "support_faq.txt")
INDEX_FILE = os.getenv("INDEX_FILE", "faq_index.json")

BATCH_SIZE = 50

client = genai.Client(api_key=GEMINI_API_KEY)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("faq_index_builder")


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text(text, chunk_size=900, overlap=120):
    """
    Split FAQ text into chunks using paragraph and sentence boundaries.
    Avoid splitting normal FAQ sections unnecessarily.
    """

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    chunks = []

    for paragraph in paragraphs:

        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)

        current_chunk = ""

        for sentence in sentences:

            if len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                start = 0

                while start < len(sentence):
                    end = start + chunk_size
                    chunks.append(sentence[start:end].strip())

                    if end >= len(sentence):
                        break

                    start = end - overlap

                continue

            candidate = (
                current_chunk + " " + sentence
            ).strip()

            if len(candidate) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())

                # Keep a small overlap from the previous chunk.
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk = candidate

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

    return chunks


# ============================================================
# VECTOR NORMALIZATION
# ============================================================

def normalize(vector):
    magnitude = sum(x * x for x in vector) ** 0.5

    if magnitude == 0:
        raise ValueError("Received a zero-length embedding.")

    return [x / magnitude for x in vector]


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(texts):
    """
    Generate Gemini retrieval-document embeddings.
    """

    response = client.models.embed_content(
        model=GEMINI_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )

    if not response.embeddings:
        raise RuntimeError("Gemini returned no embeddings.")

    if len(response.embeddings) != len(texts):
        raise RuntimeError(
            "Embedding count does not match input count."
        )

    return [
        normalize(embedding.values)
        for embedding in response.embeddings
    ]


# ============================================================
# BUILD VECTOR INDEX
# ============================================================

def build_index():

    if not os.path.isfile(FAQ_FILE):
        raise FileNotFoundError(
            f"FAQ file not found: {FAQ_FILE}"
        )

    with open(FAQ_FILE, "r", encoding="utf-8") as file:
        faq_content = file.read().strip()

    if not faq_content:
        raise ValueError("FAQ file is empty.")

    chunks = split_text(faq_content)

    if not chunks:
        raise ValueError("No FAQ chunks were generated.")

    logger.info("Generated %d FAQ chunks.", len(chunks))

    index = []

    for start in range(0, len(chunks), BATCH_SIZE):

        batch = chunks[start:start + BATCH_SIZE]

        logger.info(
            "Embedding batch %d/%d",
            start // BATCH_SIZE + 1,
            (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE,
        )

        embeddings = generate_embeddings(batch)

        for chunk, embedding in zip(batch, embeddings):

            index.append({
                "text": chunk,
                "embedding": embedding,
                "source": FAQ_FILE,
            })

    output = {
        "model": GEMINI_MODEL,
        "dimensions": EMBEDDING_DIMENSIONS,
        "documents": index,
    }

    with open(INDEX_FILE, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    logger.info(
        "Successfully created %s with %d vectors.",
        INDEX_FILE,
        len(index),
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    build_index()
