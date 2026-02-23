from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
import json
import math
import datetime

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── Step 1: Define tools ──────────────────────────────────
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Input must be a valid Python math expression.
    Examples: '2 + 2', 'math.sqrt(144)', '15 * 8.5'"""
    try:
        # Safe evaluation with math functions available
        result = eval(expression, {"math": math, "__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_current_date() -> str:
    """Get the current date and time."""
    now = datetime.datetime.now()
    return f"Current date: {now.strftime('%A, %B %d %Y')} Time: {now.strftime('%H:%M')}"


@tool
def search_knowledge(query: str) -> str:
    """Search a local knowledge base for information about the Netherlands and AI.
    Use this when asked about Dutch topics or AI concepts."""
    knowledge = {
        "amsterdam": "Amsterdam is the capital of the Netherlands with 800,000+ inhabitants. Known for canals, museums, and cycling.",
        "netherlands": "The Netherlands is a Western European country known for tulips, windmills, cycling, and being below sea level.",
        "python": "Python is the most popular AI programming language, known for simplicity and vast library ecosystem.",
        "langchain": "LangChain is a framework for building AI applications, providing tools for RAG, agents, and chains.",
        "openai": "OpenAI created GPT-4 and ChatGPT. Founded in 2015, now one of the leading AI companies.",
        "anthropic": "Anthropic created Claude, an AI assistant focused on safety. Founded by former OpenAI researchers.",
        "rag": "RAG (Retrieval Augmented Generation) combines vector search with LLM generation for document Q&A.",
        "stroopwafel": "Stroopwafel is a traditional Dutch treat — two thin waffles with caramel syrup in between.",
    }

    query_lower = query.lower()
    results = []
    for key, value in knowledge.items():
        if key in query_lower or query_lower in key:
            results.append(value)

    return "\n".join(results) if results else "No information found for that query."


@tool
def write_file(filename: str, content: str) -> str:
    """Write content to a file. Use this to save results, reports, or summaries."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {filename}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


# ── Step 2: The agent loop ────────────────────────────────
tools = [calculate, get_current_date, search_knowledge, write_file]
tools_map = {t.name: t for t in tools}

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are a helpful AI agent with access to tools.
When given a task:
1. Think about what tools you need
2. Use tools to gather information or perform actions
3. Combine results to complete the task
4. Always provide a clear final answer

Available tools: calculate, get_current_date, search_knowledge, write_file
Be systematic and thorough."""


def run_agent(user_request: str, verbose: bool = True) -> str:
    """Run the agent loop until task is complete."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_request}
    ]

    if verbose:
        print(f"\n🎯 Task: {user_request}")
        print("=" * 50)

    step = 0
    while True:
        step += 1
        if verbose:
            print(f"\n🤔 Step {step}: Thinking...")

        # Ask LLM what to do next
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Did the LLM want to use tools?
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if verbose:
                    print(f"🔧 Using tool: {tool_name}({tool_args})")

                # Execute the tool
                tool_fn = tools_map[tool_name]
                result = tool_fn.invoke(tool_args)

                if verbose:
                    print(f"📤 Result: {result}")

                # Add tool result to messages
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                ))
        else:
            # No tool calls — agent is done
            final_answer = response.content
            if verbose:
                print(f"\n✅ Final Answer:\n{final_answer}")
            return final_answer


# ── Step 3: Test the agent ────────────────────────────────
print("=" * 60)
print("AI AGENT — AUTONOMOUS TASK COMPLETION")
print("=" * 60)

tasks = [
    "What is today's date and what day of the week is it?",

    "Calculate the total cost: 3 items at €12.99 each, "
    "plus 2 items at €8.50 each. Apply a 15% discount.",

    "Research Amsterdam and the Netherlands, then write a "
    "short travel guide to the file 'travel_guide.txt'",
]

for task in tasks:
    run_agent(task)
    print("\n" + "=" * 60)

# ── Step 4: Interactive mode ──────────────────────────────
print("\nInteractive Agent — type 'quit' to exit")
print("=" * 60)

while True:
    user_input = input("\nYour task: ").strip()
    if user_input.lower() == "quit":
        break
    if not user_input:
        continue
    run_agent(user_input)