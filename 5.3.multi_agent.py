from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from ddgs import DDGS
import datetime

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── Shared tools ──────────────────────────────────────────
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
            f"[{i+1}] {r['title']}\n    {r['body']}\n    URL: {r['href']}"
            for i, r in enumerate(results)
        ])
    except Exception as e:
        return f"Search error: {str(e)}"


@tool
def write_file(filename: str, content: str) -> str:
    """Save content to a file."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Saved {len(content)} characters to {filename}"


# ── Agent runner ──────────────────────────────────────────
def run_agent(
    name: str,
    system_prompt: str,
    task: str,
    tools: list,
    max_steps: int = 8
) -> str:
    """Generic agent loop — reusable for any agent."""
    tools_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task}
    ]

    print(f"\n{'='*55}")
    print(f"🤖 {name} starting...")
    print(f"{'='*55}")

    for step in range(max_steps):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                args = tool_call["args"]

                if tool_name == "search_web":
                    print(f"  🌐 Searching: '{args.get('query', '')}'")
                elif tool_name == "write_file":
                    print(f"  📝 Writing: {args.get('filename', '')}")
                else:
                    print(f"  🔧 Tool: {tool_name}")

                result = tools_map[tool_name].invoke(args)
                preview = str(result)[:150] + "..." \
                    if len(str(result)) > 150 else str(result)
                print(f"     → {preview}")

                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                ))
        else:
            print(f"\n✅ {name} done!")
            return response.content

    return "Max steps reached."


# ── Agent 1: Researcher ───────────────────────────────────
RESEARCHER_PROMPT = """You are an expert research analyst.
Your job is to gather comprehensive, factual information on a topic.

Instructions:
- Search for the topic from multiple angles
- Use 3-5 different search queries to cover the topic thoroughly
- Gather facts, statistics, trends, and expert opinions
- Return a detailed structured summary of ALL findings
- Do NOT write a final report — just return raw research findings
- Be thorough and include specific details, numbers, and examples"""


def researcher(topic: str) -> str:
    """Agent 1: Researches a topic and returns raw findings."""
    return run_agent(
        name="Researcher",
        system_prompt=RESEARCHER_PROMPT,
        task=f"Research this topic thoroughly: {topic}",
        tools=[search_web]
    )


# ── Agent 2: Writer ───────────────────────────────────────
WRITER_PROMPT = """You are an expert report writer and analyst.
Your job is to transform raw research into a polished, professional report.

Instructions:
- Take the research findings provided and structure them clearly
- Write in a professional but accessible tone
- Use headers, bullet points, and clear sections
- Include an executive summary at the top
- Add a conclusion with key takeaways
- Save the final report to a file
- The filename should reflect the topic"""


def writer(topic: str, research_findings: str) -> str:
    """Agent 2: Transforms research findings into a polished report."""
    task = f"""Transform these research findings into a polished report about: {topic}

RESEARCH FINDINGS:
{research_findings}

Write a professional report and save it to a file."""

    return run_agent(
        name="Writer",
        system_prompt=WRITER_PROMPT,
        task=task,
        tools=[write_file]
    )


# ── Orchestrator ──────────────────────────────────────────
def research_and_write(topic: str):
    """Orchestrate Researcher + Writer to produce a polished report."""
    print(f"\n{'#'*55}")
    print(f"# MULTI-AGENT RESEARCH SYSTEM")
    print(f"# Topic: {topic}")
    print(f"{'#'*55}")

    # Step 1: Researcher gathers information
    print("\n📚 PHASE 1: Research")
    findings = researcher(topic)

    # Step 2: Writer creates the report
    print("\n✍️  PHASE 2: Writing")
    report = writer(topic, findings)

    print(f"\n{'#'*55}")
    print("# FINAL REPORT PREVIEW")
    print(f"{'#'*55}")
    print(report[:500] + "..." if len(report) > 500 else report)

    return report


# ── Run it ────────────────────────────────────────────────
print("MULTI-AGENT RESEARCH SYSTEM")
print("Two AI agents working together: Researcher + Writer")
print("=" * 55)
print("\nWhat topic should the agents research and report on?")
print("Examples:")
print("  - The future of AI agents in 2026")
print("  - Electric vehicles market trends")
print("  - Python vs JavaScript for AI development")
print()

while True:
    topic = input("Topic: ").strip()
    if topic.lower() == "quit":
        break
    if not topic:
        continue
    research_and_write(topic)
    print("\n" + "=" * 55)
    another = input("\nResearch another topic? (yes/no): ").strip()
    if another.lower() != "yes":
        break

print("\nGoodbye!")