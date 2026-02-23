# AI Engineering Journey 🤖

A collection of practical AI projects built while learning AI Engineering.
Covers OpenAI and Anthropic APIs, prompt engineering, tool calling, and document processing.

## Projects

### Week 1 — API Foundations
| Project | Description |
|---------|-------------|
| hello_ai.py | First calls to OpenAI and Anthropic APIs |
| chatbot.py | Multi-turn chatbot with conversation memory |
| personas.py | Custom AI personas using system prompts |
| streaming_chatbot.py | Real-time streaming responses |
| structured_output.py | Extract structured JSON from messy text |

### Week 2 — Core Skills
| Project | Description |
|---------|-------------|
| prompt_engineering.py | 5 prompt engineering techniques compared |
| tool_calling.py | AI that triggers real Python functions |
| summarizer.py | PDF summarizer (short documents) |
| summarizer_long.py | PDF summarizer for documents of any length |

## Setup

1. Clone this repo
2. Install dependencies:
   pip install openai anthropic pymupdf
3. Set your API keys:
   export OPENAI_API_KEY="your-key-here"
   export ANTHROPIC_API_KEY="your-key-here"
4. Run any script:
   python week1/chatbot.py

## Tech Stack
- Python 3.14
- OpenAI API (GPT-4o-mini)
- Anthropic API (Claude)
- PyMuPDF (PDF processing)

## Progress
- [x] Week 1 — API Foundations
- [x] Week 2 — Core Skills
- [x] Week 3 — Embeddings & RAG
- [ ] Week 4 — LangChain & Deployment
- [ ] Week 5 — AI Agents
- [ ] Week 6 — Capstone Project