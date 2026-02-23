from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
import sys

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ── Step 1: Basic chat ────────────────────────────────────
print("=" * 60)
print("1. BASIC CHAT")
print("=" * 60)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Be concise."),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()
response = chain.invoke({"question": "What is LangChain in one sentence?"})
print(f"Response: {response}\n")


# ── Step 2: Prompt templates ──────────────────────────────
print("=" * 60)
print("2. PROMPT TEMPLATES")
print("=" * 60)

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {domain} teacher. Explain concepts clearly."),
    ("human", "Explain {concept} to a {audience}.")
])

explain_chain = explain_prompt | llm | StrOutputParser()

examples = [
    {"domain": "AI", "concept": "neural networks", "audience": "10 year old"},
    {"domain": "finance", "concept": "compound interest", "audience": "college student"},
]

for ex in examples:
    print(f"Explaining '{ex['concept']}' to a {ex['audience']}:")
    print(explain_chain.invoke(ex))
    print()


# ── Step 3: RAG with LangChain ────────────────────────────
print("=" * 60)
print("3. RAG WITH LANGCHAIN")
print("=" * 60)

if len(sys.argv) > 1:
    pdf_path = sys.argv[1]
else:
    pdf_path = input("Enter PDF path for RAG demo: ").strip().strip('"')

print(f"Loading {pdf_path}...")
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

print("Creating vector store...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Modern LangChain RAG chain
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer based ONLY on the context below.
If the answer is not in the context, say 'Not found in document.'
Context: {context}"""),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

questions = [
    "What are the main priorities?",
    "What is planned for housing?",
]

for question in questions:
    print(f"\nQ: {question}")
    print(f"A: {rag_chain.invoke(question)}")