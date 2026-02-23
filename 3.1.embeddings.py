import openai
import json

client = openai.OpenAI()


# ── Step 1: Get an embedding ─────────────────────────────
def get_embedding(text: str) -> list[float]:
    """Convert text to a vector of numbers."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


# ── Step 2: Calculate similarity ─────────────────────────
def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """How similar are two vectors? Returns 0 (different) to 1 (identical)."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a ** 2 for a in vec1) ** 0.5
    magnitude2 = sum(b ** 2 for b in vec2) ** 0.5
    return dot_product / (magnitude1 * magnitude2)


# ── Step 3: Find most similar sentence ───────────────────
def find_most_similar(query: str, sentences: list[str]) -> list[tuple]:
    """Find sentences most similar to the query."""
    query_embedding = get_embedding(query)

    results = []
    for sentence in sentences:
        sentence_embedding = get_embedding(sentence)
        similarity = cosine_similarity(query_embedding, sentence_embedding)
        results.append((similarity, sentence))

    # Sort by similarity score, highest first
    results.sort(reverse=True)
    return results


# ── Step 4: Test it ──────────────────────────────────────
sentences = [
    "The Netherlands has an excellent public transport system",
    "Dutch cycling infrastructure is among the best in the world",
    "Python is the most popular programming language for AI",
    "Machine learning models require large amounts of training data",
    "Amsterdam is the capital city of the Netherlands",
    "Neural networks are inspired by the human brain",
    "The weather in Holland is often cloudy and rainy",
    "OpenAI released GPT-4 in March 2023",
    "Cats and dogs are the most popular pets",
    "Electric vehicles are becoming increasingly affordable",
]

print("=" * 60)
print("SEMANTIC SEARCH DEMO")
print("=" * 60)

queries = [
    "What programming tools are used in artificial intelligence?",
    "Tell me about the Netherlands",
    "How do bikes work in Holland?",
]

for query in queries:
    print(f"\nQuery: '{query}'")
    print("Most similar sentences:")
    results = find_most_similar(query, sentences)
    for i, (score, sentence) in enumerate(results[:3]):
        bar = "█" * int(score * 20)
        print(f"  {i+1}. [{score:.3f}] {bar}")
        print(f"      {sentence}")

# ── Step 5: Interactive mode ─────────────────────────────
print("\n" + "=" * 60)
print("Try your own queries! Type 'quit' to exit.")
print("=" * 60)

while True:
    query = input("\nYour query: ").strip()
    if query.lower() == "quit":
        break
    if not query:
        continue

    results = find_most_similar(query, sentences)
    print("Most similar:")
    for i, (score, sentence) in enumerate(results[:3]):
        bar = "█" * int(score * 20)
        print(f"  {i+1}. [{score:.3f}] {bar}")
        print(f"      {sentence}")