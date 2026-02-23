from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
import sys

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)


# ── Load PDF into vector store ────────────────────────────
def load_pdf(pdf_path: str):
    print(f"Loading {pdf_path}...")
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    print("Vector store ready!\n")
    return vectorstore.as_retriever(search_kwargs={"k": 4})


# ── Prompts ───────────────────────────────────────────────

# Step 1: Rewrite the question using chat history
# so "How will they fund that?" becomes
# "How will the coalition fund the housing plans?"
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given the conversation history and a follow-up question,
rewrite the follow-up question to be fully self-contained.
If the question is already clear, return it unchanged.
Return ONLY the rewritten question, nothing else."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Follow-up question: {question}")
])

# Step 2: Answer using retrieved context
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful document analyst.
Answer questions based ONLY on the context provided.
If the answer is not in the context say: 'Not found in document.'
Be specific and cite details from the context.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ── Main chat function ────────────────────────────────────
def chat(question: str, history: list, retriever) -> str:

    # Step 1: Rewrite question with context from history
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    standalone_question = rewrite_chain.invoke({
        "history": history,
        "question": question
    })

    print(f"  [Rewritten: {standalone_question}]")

    # Step 2: Retrieve relevant chunks
    docs = retriever.invoke(standalone_question)
    context = format_docs(docs)

    # Step 3: Generate answer with full history
    answer_chain = answer_prompt | llm | StrOutputParser()
    answer = answer_chain.invoke({
        "history": history,
        "question": standalone_question,
        "context": context
    })

    return answer


# ── Main program ──────────────────────────────────────────
def main():
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("Enter PDF path: ").strip().strip('"')

    retriever = load_pdf(pdf_path)

    print("=" * 60)
    print("RAG Chatbot with Memory")
    print("Type 'quit' to exit, 'reset' to clear history")
    print("=" * 60)

    # This list grows with every turn — it IS the memory
    history = []

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            history = []
            print("History cleared.")
            continue
        if not user_input:
            continue

        answer = chat(user_input, history, retriever)
        print(f"\nAI: {answer}")

        # Add this turn to history
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=answer))


if __name__ == "__main__":
    main()