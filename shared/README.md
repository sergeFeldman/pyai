# shared

Shared foundation library for Python projects under the `pyai` monorepo.

## Contents

| Package | Description |
|---|---|
| `core` | Cross-cutting utilities: `Configurable`, `ConfigurableObjectFactory`, `Singleton`, `SerializableMixin`, `Explainable`, `ExplainableMixin` |

## Installation

Install as an editable package into your Python environment:

```bash
pip install -e path/to/shared
```

After installation, import directly by package name:

```python
import core
from core import Configurable, Singleton
```

## Usage by Project

| Project | Uses |
|---|---|
| `insurance_workflow` | `core.Configurable`, `core.ConfigurableObjectFactory`, `core.Singleton`, `core.SerializableMixin`, `core.Explainable`, `core.ExplainableMixin` |

## Documentation

- [Core Foundation](docs/core.md) — class reference and usage guide
