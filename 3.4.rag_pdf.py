import openai
import chromadb
import fitz
import os
import sys

client = openai.OpenAI()
chroma_client = chromadb.Client()


# ── Step 1: PDF text extraction ───────────────────────────
def extract_text(filepath: str) -> str:
    doc = fitz.open(filepath)
    text = ""
    for page_num, page in enumerate(doc):
        text += f"\n[Page {page_num + 1}]\n"
        text += page.get_text()
    doc.close()
    return text


# ── Step 2: Smart chunking ────────────────────────────────
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    """Split text into chunks, tracking page numbers."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Find page number for this chunk
        page_num = chunk.count("[Page ")
        last_page = chunk.rfind("[Page ")
        if last_page != -1:
            page_info = chunk[last_page:last_page + 12]
        else:
            page_info = "Page 1"

        chunks.append({
            "text": chunk,
            "page": page_info.strip()
        })
        start = end - overlap

    return chunks


# ── Step 3: Embedding functions ───────────────────────────
def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text[:8000]  # safety limit
    )
    return response.data[0].embedding


# ── Step 4: Load PDF into vector DB ──────────────────────
def load_pdf(filepath: str) -> chromadb.Collection:
    filename = os.path.basename(filepath)
    collection_name = "pdf_" + "".join(c for c in filename[:20] if c.isalnum())

    collection = chroma_client.create_collection(name=collection_name)

    print(f"Reading: {filename}")
    text = extract_text(filepath)
    chunks = chunk_text(text)
    print(f"Split into {len(chunks)} chunks")
    print("Embedding chunks (this takes a moment)...")

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk["text"])
        collection.add(
            documents=[chunk["text"]],
            embeddings=[embedding],
            metadatas=[{"page": chunk["page"], "chunk": i}],
            ids=[f"chunk_{i}"]
        )
        # Progress indicator
        if (i + 1) % 5 == 0 or i == len(chunks) - 1:
            print(f"  {i+1}/{len(chunks)} chunks loaded...")

    print(f"\nDone! {len(chunks)} chunks ready for search.\n")
    return collection


# ── Step 5: RAG search + answer ───────────────────────────
def ask(question: str, collection, show_sources: bool = True) -> str:
    # Retrieve
    query_embedding = get_embedding(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=4
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if show_sources:
        print("\n📚 Retrieved sources:")
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            print(f"  {i+1}. [{meta.get('page', '')}] {doc[:100].strip()}...")
        print()

    # Augment
    context = "\n\n---\n\n".join(docs)

    # Generate
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are an expert document analyst.
Answer questions based ONLY on the provided document context.
Always mention which part of the document supports your answer.
If the answer is not in the context say: 
'This information is not found in the document.'
Be precise and cite specific details from the context."""
            },
            {
                "role": "user",
                "content": f"""Document context:
{context}

Question: {question}

Answer based only on the document:"""
            }
        ],
        temperature=0.1
    )

    return response.choices[0].message.content


# ── Step 6: Main program ──────────────────────────────────
def main():
    # Get PDF path
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("Enter path to PDF: ").strip().strip('"')

    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    # Load PDF into vector DB
    collection = load_pdf(pdf_path)

    # Chat loop
    print("=" * 60)
    print(f"Chatting with: {os.path.basename(pdf_path)}")
    print("Type 'quiet' to hide sources, 'quit' to exit")
    print("=" * 60)

    show_sources = True

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "quit":
            print("Goodbye!")
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