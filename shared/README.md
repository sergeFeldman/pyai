# shared

Shared foundation library for Python projects under the `pyai` monorepo.

## Contents

| Package | Description |
|---|---|
| `core` | Cross-cutting utilities: `Configurable`, `ConfigurableObjectFactory`, `Singleton`, `SerializableMixin`, `Explainable`, `ExplainableMixin`, `EntityMetadata`, `KeyedRegistry` |
| `data` | Storage abstractions: `DataStorage`, `DataStorageFactory`, `JsonDataStorage`, `CsvDataStorage` |

## Installation

Install as an editable package into your Python environment:

```bash
pip install -e path/to/shared
```

After installation, import directly by package name:

```python
import shared.core as shd_core
import shared.data as shd_data
```

## Usage by Project

| Project | Uses |
|---|---|
| `insurance_workflow` | `shd_core.Configurable`, `shd_core.ConfigurableObjectFactory`, `shd_core.Singleton`, `shd_core.SerializableMixin`, `shd_core.Explainable`, `shd_core.ExplainableMixin`, `shd_core.EntityMetadata`, `shd_core.KeyedRegistry`, `shd_data.DataStorageFactory`, `shd_data.JsonDataStorage`, `shd_data.CsvDataStorage` |

## Documentation

- [Core Foundation](docs/core.md) - class reference and usage guide
- [Data Storage](docs/data.md) - storage backends and factory
