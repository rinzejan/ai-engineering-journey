import openai
import anthropic
import os

# ── OpenAI ──────────────────────────────────────
client = openai.OpenAI()  # auto-reads OPENAI_API_KEY

response = client.chat.completions.create(
    model="gpt-4o-mini",  # cheap and fast for learning
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello and tell me what you are."}
    ]
)
print("OpenAI:", response.choices[0].message.content)

# ── Anthropic (Claude) ───────────────────────────
claude = anthropic.Anthropic()  # auto-reads ANTHROPIC_API_KEY

message = claude.messages.create(
    model="claude-haiku-4-5-20251001",  # fast and affordable
    max_tokens=256,
    messages=[
        {"role": "user", "content": "Say hello and tell me what you are."}
    ]
)
print("Claude:", message.content[0].text)