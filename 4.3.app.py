import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
import tempfile
import os
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Document Chat",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Chat with your Document")
st.caption("Upload a PDF and ask questions about it")

# ── Initialize session state ──────────────────────────────
# Session state persists across reruns — this is Streamlit's memory
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# ── Load PDF function ─────────────────────────────────────
@st.cache_resource  # cache so it doesn't reload on every interaction
def load_pdf(file_path: str, file_name: str):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=f"doc_{file_name[:10]}"
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})


# ── RAG chat function ─────────────────────────────────────
def chat(question: str, history: list, retriever) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    # Rewrite question using history
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", """Given the conversation history and a follow-up question,
rewrite the follow-up to be fully self-contained.
Return ONLY the rewritten question."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Follow-up: {question}")
    ])

    standalone = (rewrite_prompt | llm | StrOutputParser()).invoke({
        "history": history,
        "question": question
    })

    # Retrieve relevant chunks
    docs = retriever.invoke(standalone)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Generate answer
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """Answer based ONLY on the context below.
If the answer is not in the context say: 'Not found in document.'
Be specific and cite details.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    answer = (answer_prompt | llm | StrOutputParser()).invoke({
        "history": history,
        "question": standalone,
        "context": context
    })

    return answer


# ── Sidebar: PDF upload ───────────────────────────────────
with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file:
        if uploaded_file.name != st.session_state.pdf_name:
            # New file uploaded — reset everything
            st.session_state.messages = []
            st.session_state.pdf_name = uploaded_file.name

            with st.spinner("Reading and indexing PDF..."):
                # Save to temp file (PyMuPDF needs a file path)
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                st.session_state.retriever = load_pdf(
                    tmp_path,
                    uploaded_file.name
                )

            st.success(f"Ready! Ask me anything about {uploaded_file.name}")

    if st.session_state.pdf_name:
        st.divider()
        st.caption(f"📄 {st.session_state.pdf_name}")

        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.caption("Built with LangChain + Streamlit")


# ── Main chat area ────────────────────────────────────────
if not st.session_state.retriever:
    st.info("👈 Upload a PDF in the sidebar to get started")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if question := st.chat_input("Ask a question about your document..."):

        # Show user message
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        # Build LangChain history format
        lc_history = []
        for msg in st.session_state.messages[:-1]:  # exclude current question
            if msg["role"] == "user":
                lc_history.append(HumanMessage(content=msg["content"]))
            else:
                lc_history.append(AIMessage(content=msg["content"]))

        # Generate and show answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chat(
                    question,
                    lc_history,
                    st.session_state.retriever
                )
            st.write(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })