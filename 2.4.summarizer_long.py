import openai
import fitz
import os
import sys

client = openai.OpenAI()


# ── Step 1: Extract text ─────────────────────────────────
def extract_text(filepath: str) -> str:
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


# ── Step 2: Split into chunks ────────────────────────────
def split_into_chunks(text: str, chunk_size: int = 8000) -> list[str]:
    """Split text into overlapping chunks so context isn't lost at boundaries."""
    chunks = []
    overlap = 500  # characters of overlap between chunks
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # overlap with next chunk

    return chunks


# ── Step 3: Summarize each chunk (MAP) ───────────────────
def summarize_chunk(chunk: str, chunk_num: int, total: int) -> str:
    """Summarize a single chunk of the document."""
    print(f"  Summarizing chunk {chunk_num}/{total}...")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a document analyst. 
Summarize this section of a larger document.
Extract the most important facts, decisions, and points.
Be concise but don't miss anything important.
Write in English even if the source is in another language."""
            },
            {
                "role": "user",
                "content": f"Summarize this document section:\n\n{chunk}"
            }
        ],
        temperature=0.2
    )
    return response.choices[0].message.content


# ── Step 4: Combine chunk summaries (REDUCE) ─────────────
def combine_summaries(summaries: list[str], style: str) -> str:
    """Combine all chunk summaries into one final summary."""

    styles = {
        "executive": """You are a senior business analyst.
Create a final executive summary from these section summaries.
Structure as:
- WHAT: What is this document? (2 sentences)
- KEY POINTS: 5-7 most important points
- ACTION ITEMS: Decisions or actions suggested
- BOTTOM LINE: One sentence conclusion""",

        "technical": """You are a senior software architect.
Create a comprehensive technical summary from these section summaries.
Focus on architecture, technologies, challenges and solutions.""",

        "simple": """You are a teacher explaining to a smart 15-year-old.
Create a clear, simple summary from these section summaries.
Use plain language, no jargon, explain why it matters."""
    }

    combined = "\n\n---\n\n".join(
        [f"Section {i+1}:\n{s}" for i, s in enumerate(summaries)]
    )

    print("\nCombining all sections into final summary...")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": styles[style]},
            {
                "role": "user",
                "content": f"Create a final summary from these section summaries:\n\n{combined}"
            }
        ],
        temperature=0.3
    )
    return response.choices[0].message.content


# ── Step 5: Main program ─────────────────────────────────
def main():
    # Get PDF path
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("Enter path to PDF file: ").strip().strip('"')

    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    # Pick style
    print("\nSummary style:")
    print("  1. Executive")
    print("  2. Technical")
    print("  3. Simple")
    choice = input("\nChoose 1, 2, or 3: ").strip()
    styles = {"1": "executive", "2": "technical", "3": "simple"}
    style = styles.get(choice, "executive")

    # Extract
    print(f"\nReading: {os.path.basename(pdf_path)}")
    text = extract_text(pdf_path)
    word_count = len(text.split())
    char_count = len(text)
    print(f"Extracted {word_count} words ({char_count:,} characters)")

    # Chunk
    chunks = split_into_chunks(text)
    print(f"Split into {len(chunks)} chunks\n")

    # Map — summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        summary = summarize_chunk(chunk, i + 1, len(chunks))
        chunk_summaries.append(summary)

    # Reduce — combine into final summary
    final_summary = combine_summaries(chunk_summaries, style)

    # Print result
    print("\n" + "=" * 60)
    print(final_summary)
    print("=" * 60)

    # Save result
    output_file = os.path.splitext(pdf_path)[0] + f"_full_summary_{style}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Full Summary of: {os.path.basename(pdf_path)}\n")
        f.write(f"Style: {style} | Words: {word_count}\n")
        f.write("=" * 60 + "\n\n")
        f.write(final_summary)

    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()