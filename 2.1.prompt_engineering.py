import openai

client = openai.OpenAI()

def ask(system: str, user: str, temperature=0.7) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    return response.choices[0].message.content


topic = "Why do some startups succeed and others fail?"

print("=" * 60)

# ── Technique 1: Basic (no engineering) ─────────────────
print("1. BASIC")
print(ask("You are a helpful assistant.", topic))
print()

# ── Technique 2: Role prompting ──────────────────────────
print("2. ROLE PROMPTING")
print(ask(
    "You are a partner at a top venture capital firm with 20 years "
    "of experience investing in startups. You have seen hundreds of "
    "companies succeed and fail up close.",
    topic
))
print()

# ── Technique 3: Few-shot examples ───────────────────────
print("3. FEW-SHOT")
print(ask(
    "You are a helpful assistant.",
    f"""Here are examples of the answer style I want:

Q: Why do people exercise?
A: Three reasons dominate: energy, vanity, and fear. Energy because \
movement creates more movement. Vanity because we care how we look. \
Fear because doctors said so.

Q: Why do people procrastinate?
A: Procrastination is not laziness — it is emotion regulation. We avoid \
tasks that make us feel anxious, bored, or incompetent. The task is not \
the problem. The feeling is.

Now answer in the exact same punchy style:
Q: {topic}
A:"""
))
print()

# ── Technique 4: Chain of thought ────────────────────────
print("4. CHAIN OF THOUGHT")
print(ask(
    "You are a business analyst.",
    f"""Answer this question by thinking step by step.
First analyze the question, then examine evidence, 
then form a conclusion.

Question: {topic}"""
))
print()

# ── Technique 5: Constrained output ─────────────────────
print("5. CONSTRAINED OUTPUT")
print(ask(
    "You are a helpful assistant.",
    f"""Answer the following question with exactly:
- One bold claim (1 sentence)
- Three supporting reasons (each max 15 words)
- One contrarian take (1 sentence)

Question: {topic}"""
))
print()