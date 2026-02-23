import chromadb
import openai
import os

client = openai.OpenAI()
chroma_client = chromadb.Client()


# ── Step 1: Custom embedding function ────────────────────
def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


# ── Step 2: Create a collection (like a table in SQL) ────
collection = chroma_client.create_collection(name="knowledge_base")


# ── Step 3: Load documents into the database ─────────────
documents = [
    # Netherlands
    "The Netherlands is a country in Western Europe known for its flat landscape, windmills, and tulips.",
    "Amsterdam is the capital of the Netherlands and home to over 800,000 people.",
    "The Dutch cycling infrastructure is among the best in the world with over 35,000 km of bike paths.",
    "Rotterdam has the largest port in Europe and is a major hub for international trade.",
    "The Netherlands has a constitutional monarchy with King Willem-Alexander as head of state.",

    # AI & Technology
    "Python is the most popular programming language for AI and machine learning.",
    "OpenAI was founded in 2015 and created GPT-4, one of the most powerful language models.",
    "Machine learning models learn patterns from large datasets to make predictions.",
    "Neural networks are computing systems inspired by biological neural networks in animal brains.",
    "Anthropic created Claude, an AI assistant focused on safety and helpfulness.",

    # Food & Culture
    "Stroopwafels are a traditional Dutch treat made of two thin waffles with caramel syrup.",
    "The Dutch are known for their directness and egalitarian culture.",
    "Heineken and Amstel are famous Dutch beer brands exported worldwide.",
    "Vincent van Gogh was a Dutch post-impressionist painter born in 1853.",
    "Dutch cuisine is simple and hearty, featuring dishes like stamppot and erwtensoep.",

    # Programming
    "JavaScript is the primary language for web development and runs in browsers.",
    "Docker containers allow developers to package applications with all dependencies.",
    "Git is a distributed version control system created by Linus Torvalds in 2005.",
    "REST APIs use HTTP methods like GET, POST, PUT and DELETE to exchange data.",
    "PostgreSQL is a powerful open source relational database management system.",
]

print(f"Loading {len(documents)} documents into vector database...")

# Add documents with embeddings
embeddings = [get_embedding(doc) for doc in documents]

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print(f"Database loaded! {collection.count()} documents stored.\n")


# ── Step 4: Search function ───────────────────────────────
def search(query: str, n_results: int = 3) -> list[dict]:
    """Search the database for documents similar to query."""
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    output = []
    for i, (doc, distance) in enumerate(zip(
        results["documents"][0],
        results["distances"][0]
    )):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to 0-1 similarity score
        similarity = 1 - (distance / 2)
        output.append({"rank": i + 1, "similarity": similarity, "document": doc})

    return output

# ── Step 5: Test searches ─────────────────────────────────
print("=" * 60)
print("VECTOR DATABASE SEARCH")
print("=" * 60)

test_queries = [
    "What is the capital of Holland?",
    "Tell me about AI companies",
    "Dutch food and drinks",
    "version control for code",
]

for query in test_queries:
    print(f"\nQuery: '{query}'")
    results = search(query)
    for r in results:
        bar = "█" * int(r["similarity"] * 20)
        print(f"  {r['rank']}. [{r['similarity']:.3f}] {bar}")
        print(f"      {r['document'][:80]}...")


# ── Step 6: Interactive search ────────────────────────────
print("\n" + "=" * 60)
print("Interactive search — type 'quit' to exit")
print("=" * 60)

while True:
    query = input("\nSearch: ").strip()
    if query.lower() == "quit":
        break
    if not query:
        continue

    results = search(query, n_results=3)
    for r in results:
        bar = "█" * int(r["similarity"] * 20)
        print(f"  {r['rank']}. [{r['similarity']:.3f}] {bar}")
        print(f"      {r['document']}")