import openai

client = openai.OpenAI()

# This list IS the memory — we keep adding to it
conversation_history = [
    {
        "role": "system",
        "content": "You are a helpful AI tutor. The user is an experienced developer learning AI engineering. \
If the user hasn't specified something, ask a clarifying question rather than guessing."
    }
]

print("Chatbot ready! Type 'quit' to exit.\n")

while True:
    # Get user input
    user_input = input("You: ").strip()
    
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    
    if not user_input:
        continue

    # Add user message to history
    conversation_history.append({
        "role": "user",
        "content": user_input
    })

    # Send full history to API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation_history
    )

    # Extract reply
    reply = response.choices[0].message.content

    # Add AI reply to history (so it remembers it next turn)
    conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    print(f"AI: {reply}\n")