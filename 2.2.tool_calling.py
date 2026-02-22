import openai
import json
from datetime import datetime

client = openai.OpenAI()

# ── Step 1: Define your real Python functions ────────────
def calculate(numbers: list[float], operation: str) -> float:
    """Perform math operations reliably — no AI arithmetic!"""
    if operation == "sum":
        return sum(numbers)
    elif operation == "average":
        return sum(numbers) / len(numbers)
    elif operation == "multiply":
        result = 1
        for n in numbers:
            result *= n
        return result
    return 0

def get_current_time(city: str) -> str:
    """Return current time (simplified — no real timezone API needed)"""
    now = datetime.now().strftime("%H:%M")
    return f"Current time in {city}: {now} (local server time)"

def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert between currencies using fixed rates for demo"""
    rates = {
        ("EUR", "USD"): 1.08,
        ("USD", "EUR"): 0.93,
        ("EUR", "GBP"): 0.86,
        ("GBP", "EUR"): 1.16,
        ("USD", "GBP"): 0.79,
        ("GBP", "USD"): 1.27,
    }
    rate = rates.get((from_currency.upper(), to_currency.upper()))
    if not rate:
        return f"Conversion from {from_currency} to {to_currency} not supported"
    result = amount * rate
    return f"{amount} {from_currency} = {result:.2f} {to_currency}"


# ── Step 2: Describe tools to the AI ────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform math calculations. Always use this for any arithmetic — never calculate yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of numbers to calculate with"
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["sum", "average", "multiply"],
                        "description": "The operation to perform"
                    }
                },
                "required": ["numbers", "operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time in a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert an amount from one currency to another",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from_currency": {"type": "string"},
                    "to_currency": {"type": "string"}
                },
                "required": ["amount", "from_currency", "to_currency"]
            }
        }
    }
]


# ── Step 3: The tool execution engine ────────────────────
def run_tool(name: str, args: dict):
    """Execute whichever tool the AI requested."""
    if name == "calculate":
        return calculate(**args)
    elif name == "get_current_time":
        return get_current_time(**args)
    elif name == "convert_currency":
        return convert_currency(**args)
    return "Tool not found"


# ── Step 4: Chat loop with tool support ──────────────────
history = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Use tools whenever you need "
                   "to calculate, check time, or convert currency. "
                   "Never do math in your head."
    }
]

print("Tool-powered Chatbot! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "quit":
        break
    if not user_input:
        continue

    history.append({"role": "user", "content": user_input})

    # Send to AI with tools available
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
        tools=tools
    )

    message = response.choices[0].message

    # ── Did the AI want to call a tool? ──────────────────
    while message.tool_calls:
        history.append(message)  # add AI's tool request to history

        # Execute each tool the AI requested
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            print(f"  [calling tool: {name}({args})]")
            result = run_tool(name, args)

            # Send tool result back to AI
            history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

        # Let AI respond now that it has the tool result
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history,
            tools=tools
        )
        message = response.choices[0].message

    # Final response from AI
    reply = message.content
    history.append({"role": "assistant", "content": reply})
    print(f"AI: {reply}\n")