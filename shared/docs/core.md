# Core Foundation

The `core` package provides cross-cutting utilities used across all `pyai` projects.
It has no domain-specific dependencies — only Python stdlib and `pydantic`.

---

## Singleton

`Singleton` is a metaclass that ensures only one instance of a class exists per process.

```python
class WorkflowOrchestrator(metaclass=core.Singleton):
    ...

a = WorkflowOrchestrator(...)
b = WorkflowOrchestrator()
assert a is b  # True — same instance returned
```

`singleton()` is a function-based alternative that raises `ValueError` if instantiated more than once.

**Used by:** `WorkflowOrchestrator`, `AgentFactory`, `DataStorageFactory`

---

## Configurable

`Configurable[T]` is an abstract base class for objects initialized from a validated config object.

Subclasses declare `_config_data_type` — the `ConfigurableObjectFactory` uses this type to
validate and convert an input dictionary into a strongly typed config before construction.

```python
class ClaimAgent(Configurable[ClaimAgentConfig]):
    _config_data_type = ClaimAgentConfig

    def __init__(self, config: ClaimAgentConfig):
        super().__init__(config)
```

The validated config is accessible via the `config` property.

**Used by:** all agents, all MCP clients

---

## ConfigurableObjectFactory

`ConfigurableObjectFactory` is a singleton factory base class for creating and caching `Configurable` objects.

Subclasses declare `_TYPES_MAPPING` — a dict mapping string identifiers to `Configurable` subclasses.
The factory converts an input `dict` into the target config type and caches the resulting instance.

```python
class AgentFactory(ConfigurableObjectFactory):
    _TYPES_MAPPING = {
        "claim": ClaimAgent,
        "customer": CustomerAgent,
    }

agent = AgentFactory().get_obj("claim", config_dict)
```

**`comprehensive_hashing`** — when multiple instances of the same type share the same identifier
but differ by config content (e.g. multiple CSV storages), pass `comprehensive_hashing=True`
to the superclass `__init__`. Without it, all instances collapse to the same cache slot.

```python
class DataStorageFactory(ConfigurableObjectFactory):
    def __init__(self):
        super().__init__(comprehensive_hashing=True)
```

**Used by:** `AgentFactory`, `DataStorageFactory`

---

## SerializableMixin

`SerializableMixin` adds `to_dict()` to any dataclass. Enum fields are serialized to their `.value`;
`bool` fields are serialized to lowercase strings (`"true"` / `"false"`).

```python
@dataclass
class Claim(SerializableMixin):
    status: ClaimStatus
    is_fraud: bool

claim.to_dict()  # {"status": "denied", "is_fraud": "false", ...}
```

**Used by:** `Claim`, `Customer`, `PolicyRule` — for MCP tool return values

---

## Explainable / ExplainableMixin

`Explainable` is a marker applied via `Annotated` to dataclass fields that the explanation
engine is allowed to explain. `ExplainableMixin` exposes `explainable_attributes()` to
derive the set of explainable field names at runtime.

```python
@dataclass
class Claim(ExplainableMixin):
    status: Annotated[ClaimStatus, Explainable()]
    is_fraud: Annotated[bool, Explainable()]
    amount: float  # not explainable

Claim.explainable_attributes()  # {"status", "is_fraud"}
```

The explanation route uses this to validate that requested attributes are eligible before
passing them to the LLM agent.

**Used by:** `Claim`
