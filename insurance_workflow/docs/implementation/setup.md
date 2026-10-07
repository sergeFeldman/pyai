# Setup and Running

## Prerequisites

- Python 3.11+
- An API key for the LLM provider set in `config/agents.yaml` (Gemini, Groq, or Anthropic; Ollama runs locally without a key)
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
| `GOOGLE_API_KEY` | Yes (if using Gemini) | API key for the configured LLM provider |
| `GROQ_API_KEY` | Yes (if using Groq) | API key for the configured LLM provider |
| `ANTHROPIC_API_KEY` | Yes (if using Anthropic) | API key for the configured LLM provider |
| `LANGCHAIN_API_KEY` | Yes | LangSmith API key |
| `LANGCHAIN_TRACING_V2` | Optional | Set to `true` to enable LangSmith tracing |
| `LANGCHAIN_PROJECT` | Optional | LangSmith project name |

The active LLM provider is set in `config/agents.yaml`: whichever provider is configured there is the one that needs its API key in `.env`.

## LangSmith Tracing (Optional)

[LangSmith](https://smith.langchain.com) provides full observability for LangChain agent runs (reasoning steps, tool calls, token usage, and latency) visualized in a UI.

1. Sign up at [smith.langchain.com](https://smith.langchain.com) and create a project named `insurance_workflow`
2. Copy your API key from the LangSmith dashboard
3. Set the following in `.env`:

```bash
LANGCHAIN_API_KEY=your_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=insurance_workflow
```

4. Run Use Case 2 (`POST /claim-explanation`) and open the LangSmith project to see the full ReAct trace

No code changes are required; tracing is activated by the env vars alone.

---

## LLM Provider Configuration

Edit `config/agents.yaml` to switch providers:

```yaml
claim_explanation:
  llm_provider: gemini        # gemini | groq | anthropic | ollama
  model: gemini-3.5-flash-lite
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

**Claim Appeal Eligibility:**
```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/claim-appeal" `
  -ContentType "application/json" `
  -Body '{"message": "claim_7", "user_id": "user_1", "session_id": "session_1"}'
```

Use `http://localhost:8000/docs` for interactive Swagger UI.

---

## Rules Dashboard

The rule registry dashboard visualizes active rules and execution history for each domain.

```
http://localhost:8000/dashboard
```

Three tabs: **Rules** (DAG visualization and rule detail), **Executions** (audit history and execution trace), **Health** (graph validation findings). All share the domain selector in the header.

See [docs/architecture/dashboard.md](../architecture/dashboard.md) for a full description of each tab, the DAG overlay, finding types, and the APIs each tab consumes.

---

## Rule ETL

The ETL pipeline reads raw rule definitions from `data/in/`, detects changes, bumps versions for modified rules, and writes full version history to `data/out/`.

Run all configured domains:

```bash
PYTHONPATH=src python src/etl/rule_etl.py
```

Domains and file paths are configured in `config/etl.yaml`. The output files in `data/out/` are the source of truth loaded by the application at startup; run ETL after editing any rules in `data/in/`.
