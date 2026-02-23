import openai
import chromadb

client = openai.OpenAI()
chroma_client = chromadb.Client()

# ── Step 1: Setup vector database ────────────────────────
def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


collection = chroma_client.create_collection(name="rag_knowledge")


def add_documents(docs: list[str]):
    print(f"Loading {len(docs)} documents into knowledge base...")
    embeddings = [get_embedding(doc) for doc in docs]
    collection.add(
        documents=docs,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(docs))]
    )
    print("Knowledge base ready!\n")


def retrieve(query: str, n_results: int = 3) -> list[str]:
    """Find most relevant documents for a query."""
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results["documents"][0]


# ── Step 2: RAG answer function ───────────────────────────
def ask(question: str, show_sources: bool = True) -> str:
    """Answer a question using retrieved documents as context."""

    # RETRIEVE — find relevant documents
    relevant_docs = retrieve(question)

    if show_sources:
        print("\n📚 Retrieved sources:")
        for i, doc in enumerate(relevant_docs):
            print(f"  {i+1}. {doc}")
        print()

    # AUGMENT — build context from retrieved docs
    context = "\n".join([f"- {doc}" for doc in relevant_docs])

    # GENERATE — ask AI to answer using context
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant that answers questions
based ONLY on the provided context. 
If the answer is not in the context, say 
'I don't have information about that in my knowledge base.'
Never make up information that isn't in the context."""
            },
            {
                "role": "user",
                "content": f"""Context:
{context}

Question: {question}

Answer based only on the context above:"""
            }
        ],
        temperature=0.1  # low temperature for factual answers
    )

    return response.choices[0].message.content


# ── Step 3: Load your knowledge base ─────────────────────
knowledge_base = [
    # Netherlands
    "The Netherlands is a country in Western Europe known for its flat landscape, windmills, and tulips.",
    "Amsterdam is the capital of the Netherlands and home to over 800,000 people.",
    "The Dutch cycling infrastructure is among the best in the world with over 35,000 km of bike paths.",
    "Rotterdam has the largest port in Europe and is a major hub for international trade.",
    "The Netherlands has a constitutional monarchy with King Willem-Alexander as head of state.",
    "The Dutch economy is one of the largest in Europe, driven by trade, agriculture, and technology.",
    "The Netherlands was a founding member of the European Union and NATO.",

    # AI & Technology
    "Python is the most popular programming language for AI and machine learning.",
    "OpenAI was founded in 2015 and created GPT-4, one of the most powerful language models.",
    "Machine learning models learn patterns from large datasets to make predictions.",
    "Neural networks are computing systems inspired by biological neural networks in animal brains.",
    "Anthropic created Claude, an AI assistant focused on safety and helpfulness.",
    "Large language models like GPT-4 are trained on billions of text documents from the internet.",
    "RAG stands for Retrieval Augmented Generation — combining search with AI generation.",

    # Food & Culture
    "Stroopwafels are a traditional Dutch treat made of two thin waffles with caramel syrup.",
    "The Dutch are known for their directness and egalitarian culture.",
    "Heineken and Amstel are famous Dutch beer brands exported worldwide.",
    "Vincent van Gogh was a Dutch post-impressionist painter born in 1853 in Zundert.",
    "Dutch cuisine is simple and hearty, featuring dishes like stamppot and erwtensoep.",
    "The Rijksmuseum in Amsterdam houses masterpieces by Rembrandt and Vermeer.",

    # Programming
    "JavaScript is the primary language for web development and runs in browsers.",
    "Docker containers allow developers to package applications with all dependencies.",
    "Git is a distributed version control system created by Linus Torvalds in 2005.",
    "REST APIs use HTTP methods like GET, POST, PUT and DELETE to exchange data.",
    "PostgreSQL is a powerful open source relational database management system.",
]

add_documents(knowledge_base)


# ── Step 4: Interactive RAG chatbot ──────────────────────
print("=" * 60)
print("RAG Chatbot — answers from your knowledge base")
print("Type 'quiet' to hide sources, 'sources' to show them")
print("Type 'quit' to exit")
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

    answer = ask(user_input, show_sources=show_sources)
    print(f"AI: {answer}")