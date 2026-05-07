# Setup and Running

## Prerequisites

- Python 3.11+
- A Groq API key (or Anthropic/Ollama; see `config/agents.yaml`)
- A LangSmith API key (optional, for tracing)

## Installation

```bash
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes (if using Groq) | API key for the configured LLM provider |
| `ANTHROPIC_API_KEY` | Yes (if using Anthropic) | API key for the configured LLM provider |
| `LANGCHAIN_API_KEY` | Yes | LangSmith API key |
| `LANGCHAIN_TRACING_V2` | Optional | Set to `true` to enable LangSmith tracing |
| `LANGCHAIN_PROJECT` | Optional | LangSmith project name |

The active LLM provider is set in `config/agents.yaml`: whichever provider is configured there is the one that needs its API key in `.env`.

## LLM Provider Configuration

Edit `config/agents.yaml` to switch providers:

```yaml
claim_explanation:
  llm_provider: groq          # groq | anthropic | ollama
  model: llama-3.3-70b-versatile
  prompt_name: hwchase17/structured-chat-agent
```

## Running the App

```bash
python -m uvicorn app.main:app
```

Or via VS Code: open `.vscode/launch.json` and press F5.

## API Examples

**Claim Status:**
```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/claim-status" `
  -ContentType "application/json" `
  -Body '{"message": "claim_100", "user_id": "user_1", "session_id": "session_1"}'
```

**Claim Explanation:**
```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/claim-explanation" `
  -ContentType "application/json" `
  -Body '{"message": "claim_201", "attributes": ["status", "is_fraud"], "user_id": "user_1", "session_id": "session_1"}'
```

Use `http://localhost:8000/docs` for interactive Swagger UI.
