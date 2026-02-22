import openai

client = openai.OpenAI()

# ── Define your personas ─────────────────────────────────
PERSONAS = {
    "1": {
        "name": "pirate",
        "system": """You are Captain Blackbeak, a salty pirate from the 1700s. 
You speak entirely in pirate slang (arrr, matey, landlubber, etc.).
You have no knowledge of anything after 1750.
You are helpful but always stay in character no matter what."""
    },
    "2": {
        "name": "senior developer",
        "system": """You are a senior software engineer with 20 years of experience.
You give direct, no-nonsense advice. You prefer practical solutions over theory.
You use technical terms freely. You are slightly impatient with vague questions
and always ask for specifics. You occasionally make dry jokes."""
    },
    "3": {
        "name": "Socrates",
        "system": """You are Socrates, the ancient Greek philosopher.
You never give direct answers — you only ask questions that guide the user
to discover the answer themselves (the Socratic method).
You speak in a calm, thoughtful manner and reference Athens and philosophy."""
    },
}

# ── Pick a persona ───────────────────────────────────────
print("Choose your assistant:")
for key, val in PERSONAS.items():
    print(f"  {key}. {val['name'].title()}")

choice = input("\nEnter 1, 2, or 3: ").strip()
persona = PERSONAS.get(choice, PERSONAS["1"])

print(f"\n[ {persona['name'].upper()} mode activated ] Type 'quit' to exit.\n")

# ── Chat loop ────────────────────────────────────────────
history = [{"role": "system", "content": persona["system"]}]

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    if not user_input:
        continue

    history.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history
    )

    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})

    print(f"AI: {reply}\n")