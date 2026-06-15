# Data Storage

The `data` package provides storage abstractions for reading and writing domain objects
to different backends. All backends share the same interface; callers use `DataStorageFactory`
to get a backend by id without depending on concrete classes.

---

## DataStorageId

`DataStorageId` is an enum of supported backend identifiers passed to `DataStorageFactory.get_obj()`.

| Value | Backend |
|---|---|
| `DataStorageId.JSON` | JSON file |
| `DataStorageId.CSV` | CSV file |
| `DataStorageId.API` | API (reserved) |
| `DataStorageId.DB` | Database (reserved) |

---

## DataStorageFactory

`DataStorageFactory` is a singleton factory that creates and caches `DataStorage` instances.
Uses `comprehensive_hashing=True` so multiple storages of the same backend type but different
file paths are cached as distinct instances.

```python
storage = shd_data.DataStorageFactory().get_obj(
    shd_data.DataStorageId.JSON.value,
    {"model_class": Rule, "key_field": "id", "file_path": "data/out/claim_appeal_rules.json"}
)
```

---

## DataStorage

`DataStorage[T]` is the abstract base class for all backends. Concrete subclasses implement
three methods against their specific storage medium.

| Method | Description |
|---|---|
| `read()` | Return all objects from the backend |
| `read_by_key(key)` | Return a single object by primary key, or `None` |
| `write(objects)` | Persist all objects to the backend (full overwrite) |

The `model_class` property exposes the concrete model type the storage deserializes into.

---

## JsonDataStorage

`JsonDataStorage` reads and writes domain objects to a JSON file. Handles nested dataclasses
(e.g. `EntityMetadata` inside `DecisionRule`) by recursively deserializing dicts on read.

Config fields:

| Field | Description |
|---|---|
| `model_class` | Dataclass type to deserialize records into |
| `file_path` | Path to the JSON file (relative to working directory) |
| `key_field` | Primary key field name. Defaults to `{model_type}_id` if empty |

```python
storage = cast(shd_data.JsonDataStorage, shd_data.DataStorageFactory().get_obj(
    shd_data.DataStorageId.JSON.value,
    {"model_class": Rule, "key_field": "id", "file_path": "data/out/claim_appeal_rules.json"}
))

rules = storage.read()             # list[Rule] - empty list if file missing
rule  = storage.read_by_key("ca_min_amount")  # Rule | None
raw   = storage.read_as_dicts()    # list[dict] - no deserialization, used by ETL input
storage.write(rules)               # creates parent dirs, overwrites file
```

`read_as_dicts()` is used by the ETL pipeline to read raw input that does not yet carry
`domain` or `metadata` fields - it returns plain dicts without model construction.

`read()` and `read_as_dicts()` both return `[]` when the file does not exist rather than
raising `FileNotFoundError`. This lets the ETL treat a missing output file as an empty
registry on first run.

---

## CsvDataStorage

`CsvDataStorage` reads and writes domain objects to a CSV file. CSV headers must match
the dataclass field names exactly; any mismatch raises `ValueError` on read.

Config fields:

| Field | Description |
|---|---|
| `model_class` | Dataclass type to deserialize rows into |
| `file_path` | Path to the CSV file |

Primary key is derived by convention: `{model_type}_id` (e.g. `PolicyRule` -> `policy_rule_id`).

```python
storage = shd_data.DataStorageFactory().get_obj(
    shd_data.DataStorageId.CSV.value,
    {"model_class": PolicyRule, "file_path": "data/policy_rules.csv"}
)

rules = storage.read()           # list[PolicyRule]
rule  = storage.read_by_key("pr_ac_fraud")  # PolicyRule | None
storage.write(rules)             # creates parent dirs, overwrites file
```

Unlike `JsonDataStorage`, `CsvDataStorage.read()` raises `FileNotFoundError` when the file
is missing - CSV files are expected to be pre-populated, not generated on first run.
