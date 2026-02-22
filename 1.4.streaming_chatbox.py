import openai

client = openai.OpenAI()

history = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Be conversational and friendly."
    }
]

print("Streaming Chatbot ready! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    if not user_input:
        continue

    history.append({"role": "user", "content": user_input})

    # stream=True is the only change from before
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
        stream=True
    )

    # Print each chunk as it arrives
    print("AI: ", end="", flush=True)
    full_reply = ""

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            full_reply += delta

    print("\n")  # newline after response is complete

    # Save the full reply to history as normal
    history.append({"role": "assistant", "content": full_reply})