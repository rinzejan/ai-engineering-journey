import openai
import chromadb
import fitz
import os
import sys

client = openai.OpenAI()
chroma_client = chromadb.Client()


# ── Embedding ─────────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text[:8000]
    )
    return response.data[0].embedding


# ── Improvement 1: Query Expansion ───────────────────────
def expand_query(question: str) -> list[str]:
    """Generate multiple versions of the question for better retrieval."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Generate 3 different versions of the user's question 
to improve document retrieval. Each version should approach 
the topic from a different angle.
Return ONLY the 3 questions, one per line, no numbering."""
            },
            {"role": "user", "content": question}
        ],
        temperature=0.7
    )
    variations = response.choices[0].message.content.strip().split("\n")
    all_queries = [question] + [v.strip() for v in variations if v.strip()]
    return all_queries[:4]  # original + 3 variations


# ── Improvement 2: Reranking ──────────────────────────────
def rerank(question: str, chunks: list[str]) -> list[str]:
    """Score chunks by relevance and return reranked list."""
    if len(chunks) <= 1:
        return chunks

    chunks_text = "\n\n".join(
        [f"[{i+1}] {chunk[:300]}" for i, chunk in enumerate(chunks)]
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a relevance scorer.
Given a question and document chunks, return ONLY the numbers 
of the chunks ranked from most to least relevant.
Format: comma-separated numbers like: 3,1,4,2"""
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nChunks:\n{chunks_text}"
            }
        ],
        temperature=0
    )

    try:
        ranking = response.choices[0].message.content.strip()
        indices = [int(x.strip()) - 1 for x in ranking.split(",")]
        return [chunks[i] for i in indices if 0 <= i < len(chunks)]
    except:
        return chunks  # fallback to original order


# ── PDF loading ───────────────────────────────────────────
def extract_text(filepath: str) -> str:
    doc = fitz.open(filepath)
    text = ""
    for page_num, page in enumerate(doc):
        text += f"\n[Page {page_num + 1}]\n"
        text += page.get_text()
    doc.close()
    return text


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def load_pdf(filepath: str) -> chromadb.Collection:
    filename = os.path.basename(filepath)
    col_name = "imp_" + "".join(c for c in filename[:15] if c.isalnum())
    collection = chroma_client.create_collection(name=col_name)

    print(f"Loading: {filename}")
    text = extract_text(filepath)
    chunks = chunk_text(text)
    print(f"Embedding {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[get_embedding(chunk)],
            ids=[f"chunk_{i}"]
        )
        if (i + 1) % 10 == 0 or i == len(chunks) - 1:
            print(f"  {i+1}/{len(chunks)} done...")

    print(f"Ready!\n")
    return collection


# ── Improved RAG answer ───────────────────────────────────
def ask(question: str, collection, show_sources: bool = True) -> str:

    # Step 1: Expand query into multiple versions
    queries = expand_query(question)
    if show_sources:
        print(f"\n🔍 Query expanded into {len(queries)} versions:")
        for q in queries:
            print(f"   - {q}")

    # Step 2: Retrieve chunks for each query version
    all_chunks = []
    seen = set()
    for query in queries:
        embedding = get_embedding(query)
        results = collection.query(query_embeddings=[embedding], n_results=3)
        for doc in results["documents"][0]:
            if doc not in seen:
                all_chunks.append(doc)
                seen.add(doc)

    # Step 3: Rerank all retrieved chunks
    reranked = rerank(question, all_chunks)

    # Step 4: Use top chunks as context
    top_chunks = reranked[:4]

    if show_sources:
        print(f"\n📚 Top {len(top_chunks)} chunks after reranking:")
        for i, chunk in enumerate(top_chunks):
            print(f"  {i+1}. {chunk[:100].strip()}...")
        print()

    context = "\n\n---\n\n".join(top_chunks)

    # Step 5: Generate answer
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are an expert document analyst.
Answer questions based ONLY on the provided context.
Be specific and cite details from the context.
If the answer is not in the context say: 
'This information is not found in the document.'"""
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        temperature=0.1
    )

    return response.choices[0].message.content


# ── Main ──────────────────────────────────────────────────
def main():
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("Enter path to PDF: ").strip().strip('"')

    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    collection = load_pdf(pdf_path)

    print("=" * 60)
    print(f"Improved RAG — {os.path.basename(pdf_path)}")
    print("Type 'quiet' to hide sources, 'quit' to exit")
    print("=" * 60)

    show_sources = True

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "quiet":
            show_sources = False
            print("Sources hidden.")
            continue
        if user_input.lower() == "sources":
            show_sources = True
            print("Sources visible.")
            continue
        if not user_input:
            continue

        answer = ask(user_input, collection, show_sources)
        print(f"\nAI: {answer}")


if __name__ == "__main__":
    main()