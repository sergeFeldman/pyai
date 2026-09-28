---
name: shared-foundation
description: Contracts of the ../shared foundation library used by insurance_workflow - Configurable, ConfigurableObjectFactory, the Singleton metaclass, KeyedRegistry, SerializableMixin, Explainable and ExplainableMixin, EntityMetadata, ExecutionMetadata, and the DataStorage backends (CSV, JSON, JSONL) with DataStorageFactory. Use when creating any configurable component or factory, making a class a singleton, grouping items in a registry, serializing a dataclass, adding a storage backend, or changing anything under ../shared/src, and when debugging config validation errors, stale singleton state, or factory cache collisions.
---

# Shared Foundation

## Responsibility

Owns the contracts of `../shared/src/shared/core` and `../shared/src/shared/data`: what a subclass must declare, how instances are created and cached, and the serialization and storage behavior every layer relies on.

Does not own how insurance_workflow layers use these classes; agents, MCP clients, the rule engine, and services each describe their own usage.

No other project under `pyai` currently imports `shared`. Keep its API stable anyway, and change a contract here only together with every caller in this project.

## Configurable

Every configurable component extends `shd_core.Configurable[TConfig]`.

- Declare `_config_data_type` as a class attribute: the Pydantic model (or `dict`) the factory validates input against.
- Take the validated config object as the first `__init__` argument and pass it to `super().__init__(config)`. A `None` config raises `ValueError`.
- Read config through the `config` property; never mutate it after construction.

## ConfigurableObjectFactory

Subclass it to create and cache configurable objects by string id. It uses the `Singleton` metaclass, so every subclass is a process-wide singleton.

- Declare `_TYPES_MAPPING = {"id": ConcreteClass, ...}`. An empty or missing mapping raises `ValueError` on construction. An unknown id raises `ValueError("Provided identifier is not supported: ...")`.
- `get_obj(id, config_dict)` returns the cached instance for that key, creating it on first use. `replace_with_new=True` rebuilds it.
- The cache key is the id alone by default. Pass `comprehensive_hashing=True` to `super().__init__` when several instances share an id but differ by config (for example several CSV storages); the key becomes id plus a hash of the config. Without it, the first config wins and later configs are silently ignored.
- `_create_obj` converts the dict with `convert_from_dict` and calls `ConcreteClass(config)`. Override `_create_obj_async` (used by `get_obj_async`) when construction must await, as `AgentFactory` does for LLM agents.
- `convert_from_dict` behavior:
  - Target `dict`: input returned unchanged.
  - Unknown input keys are dropped, unless `strict_convert=True`.
  - Missing required field or (strict) extra key: `KeyError`.
  - Wrong value types only: `TypeError`.
  - Anything else: `RuntimeError`.

## Singleton

`metaclass=shd_core.Singleton` returns the first instance on every later call.

- Constructor arguments after the first call are ignored. Pass configuration on first construction only, and never rely on a later call to reconfigure.
- The metaclass and all singleton caches are not thread-safe. Do not construct singletons or mutate their state from worker threads.
- Tests reset a singleton with `Singleton._instances.pop(Cls, None)`.
- The function `shd_core.singleton(obj)` (raise on second instantiation) exists but is unused; prefer the metaclass.

## KeyedRegistry

`shd_core.KeyedRegistry[T](item_type, key_field)` groups items into lists by the value of `key_field`.

- `load(items)` replaces all contents; `add(item)` appends and never replaces.
- `get_by_key(key)` returns the internal list (or a new empty list). Never mutate the returned list; copy it first.
- Subclass to add domain queries, as `RuleRegistry` does.

## Serialization and explanation mixins

- `SerializableMixin.to_dict()` recursively converts a dataclass: Enum to `.value`, bool to the strings `"true"`/`"false"`, nested dataclasses and lists recursively. Use it for persistence, tool return values, and audit snapshots. Never use it to build values that are later compared by type (the rule engine context), because bools and Enums lose their types.
- Mark a field explainable with `Annotated[T, shd_core.Explainable()]`. `ExplainableMixin.explainable_attributes()` returns the marked field names.

## Metadata

- `EntityMetadata` is composed into persisted versioned entities (rules). `created_*` is set once; `bump(updated_by)` increments `version` and sets `updated_by` and `updated_timestamp`. Only the rule ETL writes it; application code reads it.
- `ExecutionMetadata` is composed into runtime results: `trace_id`, `executed_by`, and an auto-stamped UTC `executed_timestamp`.

## Data storage

For backend behavior (CSV, JSON, JSONL), `DataStorageFactory`, and adding a new backend, read [references/data-storage.md](references/data-storage.md).

Status: partial (`DataStorageId.API` and `DataStorageId.DB` are declared but have no implementation)

## Verification

- Run `python -m pytest tests/ -q` from the insurance_workflow root; `../shared` has no test suite of its own yet. Status: planned
- After changing a contract, grep `src/` and `tests/` for every subclass and caller and update them in the same change.

## Examples

- Normal: `DataStorageFactory().get_obj("csv", {"model_class": mdl.Claim, "file_path": "data/in/claim.csv"})` creates a `CsvDataStorage` on first call and returns the same instance for the same config.
- Edge: a factory subclass without `comprehensive_hashing` asked for `"csv"` with two different file paths returns the first storage both times.
- Edge: `convert_from_dict(ClaimAgentConfig, {})` raises `KeyError` naming `claim_mcp_client_config`.
- Edge: `Claim(..., is_fraud=True).to_dict()["is_fraud"]` is the string `"true"`, not `True`.

## Background

- [../shared/docs/core.md](../../../../shared/docs/core.md)
- [../shared/docs/data.md](../../../../shared/docs/data.md)
