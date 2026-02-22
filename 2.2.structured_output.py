import openai
import json

client = openai.OpenAI()

def extract_data(text: str) -> dict:
    """Extract structured data from any messy text."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a senior data analyst with 15 years experience processing 
business documents. When extracting data, first identify all entities 
mentioned, then map them to the correct fields, then return the JSON.
Extract information from the user's text and return ONLY valid JSON.
No explanation, no markdown, no code blocks — raw JSON only.
If a field is not found, use null.

{
    "name": "full name or null",
    "date": "YYYY-MM-DD format or null",
    "amount": number or null,
    "currency": "EUR/USD/GBP or null",
    "city": "city name or null",
    "action": "what happened in one word or null"
}"""
            },
            {
                "role": "user",
                "content": text
            }
        ],
        temperature=0  # 0 = more deterministic, better for data extraction
    )

    raw = response.choices[0].message.content.strip()
    return json.loads(raw)  # convert JSON string to Python dict


# ── Test it with different inputs ────────────────────────
test_cases = [
    "John Smith called on 3rd of March, he owes us €450 and is from Amsterdam",
    "Meeting with Sarah Johnson in Berlin on December 5th, invoice $1200",
    "Quick note: Peter called, needs refund of €89.99, based in Rotterdam, date was 15 jan",
    "No useful information in this sentence at all",
]

for text in test_cases:
    print(f"Input:  {text}")
    result = extract_data(text)
    print(f"Output: {json.dumps(result, indent=2)}")
    print("-" * 60)

print("\nTry your own input:")
while True:
    text = input("Input: ").strip()
    if text.lower() == "quit":
        break
    result = extract_data(text)
    print(json.dumps(result, indent=2))
    print()