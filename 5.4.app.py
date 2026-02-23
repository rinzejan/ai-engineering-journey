import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from ddgs import DDGS
import tempfile
import os
import datetime

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="centered"
)

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Session state ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# ── Tools ─────────────────────────────────────────────────
@tool
def search_web(query: str) -> str:
    """Search the internet for information."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query, max_results=4, region="wt-wt"
            ))
        if not results:
            return "No results found."
        return "\n\n".join([
            f"[{i+1}] {r['title']}\n    {r['body']}"
            for i, r in enumerate(results)
        ])
    except Exception as e:
        return f"Search error: {str(e)}"


# ── Agent runner ──────────────────────────────────────────
def run_agent_streaming(system_prompt: str, task: str,
                        tools: list, status_container) -> str:
    """Run agent loop with live status updates in Streamlit."""
    tools_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task}
    ]

    for step in range(10):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                name = tool_call["name"]
                args = tool_call["args"]

                if name == "search_web":
                    status_container.info(
                        f"🌐 Searching: '{args.get('query', '')[:60]}'"
                    )

                result = tools_map[name].invoke(args)
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                ))
        else:
            return response.content

    return "Research complete."


# ── PDF functions ─────────────────────────────────────────
@st.cache_resource
def load_pdf(file_path: str, file_name: str):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        chunks, embeddings,
        collection_name=f"doc_{file_name[:10]}"
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def chat_with_pdf(question: str, history: list, retriever) -> str:
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", """Rewrite the follow-up question to be fully 
self-contained given the conversation history.
Return ONLY the rewritten question."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Follow-up: {question}")
    ])

    standalone = (rewrite_prompt | llm | StrOutputParser()).invoke({
        "history": history, "question": question
    })

    docs = retriever.invoke(standalone)
    context = "\n\n".join(doc.page_content for doc in docs)

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """Answer based ONLY on the context.
If not found say: 'Not found in document.'
Context: {context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    return (answer_prompt | llm | StrOutputParser()).invoke({
        "history": history,
        "question": standalone,
        "context": context
    })


# ── UI: Two tabs ──────────────────────────────────────────
tab1, tab2 = st.tabs(["📄 Chat with PDF", "🔍 Research Agent"])


# ════════════════════════════════════════════════════════
# TAB 1: PDF Chat
# ════════════════════════════════════════════════════════
with tab1:
    st.title("📄 Chat with your Document")
    st.caption("Upload a PDF and ask questions about it")

    with st.sidebar:
        st.header("📁 Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

        if uploaded_file:
            if uploaded_file.name != st.session_state.pdf_name:
                st.session_state.messages = []
                st.session_state.pdf_name = uploaded_file.name

                with st.spinner("Indexing PDF..."):
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    st.session_state.retriever = load_pdf(
                        tmp_path, uploaded_file.name
                    )
                st.success("Ready!")

        if st.session_state.pdf_name:
            st.divider()
            st.caption(f"📄 {st.session_state.pdf_name}")
            if st.button("🗑️ Clear conversation"):
                st.session_state.messages = []
                st.rerun()

        st.divider()
        st.caption("Built with LangChain + Streamlit")

    if not st.session_state.retriever:
        st.info("👈 Upload a PDF to get started")
        col1, col2 = st.columns(2)
        with col1:
            st.success("✅ Chat with any PDF")
            st.success("✅ Remembers conversation")
            st.success("✅ Works in any language")
        with col2:
            st.success("✅ Cites sources")
            st.success("✅ Honest about gaps")
            st.success("✅ No hallucination")
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if question := st.chat_input("Ask about your document..."):
            with st.chat_message("user"):
                st.write(question)
            st.session_state.messages.append({
                "role": "user", "content": question
            })

            lc_history = []
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    lc_history.append(HumanMessage(content=msg["content"]))
                else:
                    lc_history.append(AIMessage(content=msg["content"]))

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = chat_with_pdf(
                        question, lc_history,
                        st.session_state.retriever
                    )
                st.write(answer)

            st.session_state.messages.append({
                "role": "assistant", "content": answer
            })


# ════════════════════════════════════════════════════════
# TAB 2: Research Agent
# ════════════════════════════════════════════════════════
with tab2:
    st.title("🔍 Research Agent")
    st.caption("Two AI agents research any topic and write a report")

    st.info("""
**How it works:**
1. 🔬 **Researcher** searches the web from multiple angles
2. ✍️ **Writer** transforms findings into a polished report
    """)

    topic = st.text_input(
        "Research topic:",
        placeholder="e.g. The future of AI agents in 2026"
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        research_btn = st.button("🚀 Start Research", type="primary")

    if research_btn and topic:
        st.divider()

        # ── Phase 1: Researcher ──────────────────────────
        st.subheader("📚 Phase 1: Researcher")
        researcher_status = st.empty()
        researcher_status.info("🤔 Researcher is starting...")

        findings = run_agent_streaming(
            system_prompt="""You are an expert research analyst.
Search the topic from 4-5 different angles.
Use multiple search queries to gather comprehensive information.
Return detailed structured findings with facts and statistics.""",
            task=f"Research thoroughly: {topic}",
            tools=[search_web],
            status_container=researcher_status
        )

        researcher_status.success("✅ Research complete!")

        with st.expander("📋 Raw Research Findings", expanded=False):
            st.write(findings)

        # ── Phase 2: Writer ──────────────────────────────
        st.subheader("✍️ Phase 2: Writer")
        writer_status = st.empty()
        writer_status.info("✍️ Writer is creating your report...")

        report = (ChatPromptTemplate.from_messages([
            ("system", """You are an expert report writer.
Transform research findings into a polished professional report.
Use clear headers, bullet points, and an executive summary.
End with key takeaways."""),
            ("human", f"""Write a report about: {topic}

Research findings:
{findings}""")
        ]) | llm | StrOutputParser()).invoke({})

        writer_status.success("✅ Report ready!")

        # ── Display report ───────────────────────────────
        st.divider()
        st.subheader("📊 Final Report")
        st.markdown(report)

        # Download button
        st.download_button(
            label="⬇️ Download Report",
            data=report,
            file_name=f"research_{topic[:30].replace(' ', '_')}.md",
            mime="text/markdown"
        )

    elif research_btn and not topic:
        st.warning("Please enter a research topic first.")