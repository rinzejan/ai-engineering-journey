import openai
import fitz  # pymupdf
import sys
import os

client = openai.OpenAI()


# ── Step 1: Extract text from PDF ────────────────────────
def extract_text_from_pdf(filepath: str) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


# ── Step 2: Summarize the text ───────────────────────────
def summarize(text: str, style: str = "executive") -> str:
    """Summarize text in different styles."""

    styles = {
        "executive": """You are a senior business analyst.
Summarize the document for a busy executive who has 2 minutes to read.
Structure your response as:
- WHAT: What is this document about? (2 sentences max)
- KEY POINTS: 3-5 most important points (bullet points)
- ACTION ITEMS: What decisions or actions does this suggest? (if any)
- BOTTOM LINE: One sentence conclusion.""",

        "technical": """You are a senior software architect.
Summarize the technical content focusing on:
- Architecture and design decisions
- Technologies and tools mentioned
- Technical challenges and solutions
- Implementation details worth noting""",

        "simple": """You are a teacher explaining to a smart 15-year-old.
Use simple language, no jargon. Explain what this document is about
and why it matters in plain everyday language."""
    }

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": styles[style]},
            {"role": "user", "content": f"Summarize this document:\n\n{text[:12000]}"}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content


# ── Step 3: Main program ─────────────────────────────────
def main():
    # Get PDF path from user
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("Enter path to PDF file: ").strip().strip('"')

    # Check file exists
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    print(f"\nReading: {os.path.basename(pdf_path)}")
    print("Extracting text...")

    # Extract text
    text = extract_text_from_pdf(pdf_path)
    word_count = len(text.split())
    print(f"Extracted {word_count} words from PDF")

    # Pick summary style
    print("\nSummary style:")
    print("  1. Executive (structured business summary)")
    print("  2. Technical (for developers)")
    print("  3. Simple (plain language)")
    choice = input("\nChoose 1, 2, or 3: ").strip()

    styles = {"1": "executive", "2": "technical", "3": "simple"}
    style = styles.get(choice, "executive")

    print(f"\nGenerating {style} summary...\n")
    print("=" * 60)

    summary = summarize(text, style)
    print(summary)
    print("=" * 60)

    # Save to file
    output_file = os.path.splitext(pdf_path)[0] + f"_summary_{style}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Summary of: {os.path.basename(pdf_path)}\n")
        f.write(f"Style: {style}\n")
        f.write("=" * 60 + "\n\n")
        f.write(summary)

    print(f"\nSummary saved to: {output_file}")


if __name__ == "__main__":
    main()