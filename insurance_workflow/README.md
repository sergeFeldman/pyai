# Agentic Workflow Platform for a Large Multiline Insurer

## Installation

```powershell
python -m pip install -r requirements.txt
```

## API Examples

Start the app:

```powershell
$env:PYTHONPATH="src"
python -m uvicorn app.main:app --reload
```

Request a missing claim:

```powershell
curl.exe --% -X POST http://127.0.0.1:8000/claim-status -H "Content-Type: application/json" -d "{\"message\":\"claim_X\",\"user_id\":\"user_1\",\"session_id\":\"session_1\"}"
```

Request an existing claim:

```powershell
curl.exe --% -X POST http://127.0.0.1:8000/claim-status -H "Content-Type: application/json" -d "{\"message\":\"claim_100\",\"user_id\":\"user_1\",\"session_id\":\"session_1\"}"
```

## Documents
- [Architecture Overview](docs/architecture/agentic-platform-overview.md)
- [High-Level Diagram](docs/architecture/high-level-diagram.md)
- [Component Deep Dive](docs/architecture/component-deep-dive.md)
- [Workflow Patterns](docs/architecture/workflow-patterns.md)
- [Security and Compliance](docs/architecture/security-and-compliance.md)
- [Scaling and Performance](docs/architecture/scaling-and-performance.md)
- [Implementation Roadmap](docs/roadmap/implementation-roadmap.md)
- [Risk Register](docs/risks/risk-register.md)
