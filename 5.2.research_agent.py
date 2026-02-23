from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
import datetime
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── Tools ─────────────────────────────────────────────────
@tool
def search_web(query: str) -> str:
    """Search the internet for current information on any topic.
    Returns top 5 results with titles, snippets and URLs."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                max_results=5,
                region="wt-wt",    # worldwide — no regional bias
                safesearch="off",
                timelimit=None
            ))
        if not results:
            return "No results found."
        output = []
        for i, r in enumerate(results):
            output.append(
                f"[{i+1}] {r['title']}\n"
                f"    {r['body']}\n"
                f"    URL: {r['href']}"
            )
        return "\n\n".join(output)
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def get_current_date() -> str:
    """Get the current date."""
    return datetime.datetime.now().strftime("%A, %B %d %Y")


@tool
def write_report(filename: str, title: str, content: str) -> str:
    """Save a research report to a file."""
    try:
        full_content = f"# {title}\n"
        full_content += f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        full_content += "=" * 50 + "\n\n"
        full_content += content
        with open(filename, "w", encoding="utf-8") as f:
            f.write(full_content)
        return f"Report saved to {filename} ({len(full_content)} characters)"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def summarize_findings(findings: list[str], focus: str) -> str:
    """Synthesize multiple search results into a coherent summary.
    Use this after gathering information from multiple searches."""
    combined = "\n\n".join(findings)
    response = llm.invoke([
        {"role": "system", "content": f"""You are a research analyst.
Synthesize these findings into a clear, structured summary.
Focus on: {focus}
Be factual, cite key points, and highlight the most important information."""},
        {"role": "user", "content": f"Synthesize these findings:\n\n{combined}"}
    ])
    return response.content


# ── Agent loop ────────────────────────────────────────────
tools = [search_web, get_current_date, write_report, summarize_findings]
tools_map = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are an expert research agent with internet access.

When given a research task:
1. Break it into specific search queries
2. Search for each aspect separately
3. Synthesize findings using summarize_findings
4. Write a structured report using write_report
5. Present a clear conclusion

Be thorough — use multiple searches to cover different angles.
Always save your research to a file."""


def run_agent(task: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task}
    ]

    print(f"\n🎯 Research Task: {task}")
    print("=" * 55)

    step = 0
    max_steps = 15  # safety limit

    while step < max_steps:
        step += 1
        print(f"\n🤔 Step {step}: Thinking...")

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                name = tool_call["name"]
                args = tool_call["args"]

                # Show what the agent is doing
                if name == "search_web":
                    print(f"🌐 Searching: '{args.get('query', '')}'")
                elif name == "write_report":
                    print(f"📝 Writing report: {args.get('filename', '')}")
                elif name == "summarize_findings":
                    print(f"🔍 Synthesizing findings...")
                else:
                    print(f"🔧 Using tool: {name}")

                # Execute tool
                result = tools_map[name].invoke(args)

                # Show truncated result
                result_preview = str(result)[:200] + "..." \
                    if len(str(result)) > 200 else str(result)
                print(f"   → {result_preview}")

                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                ))
        else:
            # Agent is done
            print(f"\n✅ Research complete!\n")
            print(response.content)
            return response.content

    return "Max steps reached."


# ── Test tasks ────────────────────────────────────────────
print("=" * 55)
print("RESEARCH AGENT — LIVE INTERNET ACCESS")
print("=" * 55)

# Interactive mode
print("\nWhat would you like me to research?")
print("Examples:")
print("  - Research the latest developments in AI agents")
print("  - What are the best Python libraries for data science in 2026?")
print("  - Research electric vehicles market trends")
print("  - What is happening in Dutch politics right now?")
print("\nType 'quit' to exit\n")

while True:
    task = input("Research task: ").strip()
    if task.lower() == "quit":
        break
    if not task:
        continue
    run_agent(task)
    print("\n" + "=" * 55)