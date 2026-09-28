# Data Storage Backends

Read this when reading or writing domain data through `shared.data`, adding a storage backend, or debugging a storage error.

## Common contract

- Every backend extends `DataStorage[TConfig]` (a `Configurable`) and implements `read()`, `read_by_key(key)`, and `write(objects)`.
- Every config extends `DataStorageConfig`, which requires `model_class` (the dataclass records deserialize into) and forbids extra keys.
- Obtain storages through `DataStorageFactory().get_obj(storage_id, config_dict)`. The factory maps `"csv"`, `"json"`, and `"jsonl"` to their classes and uses comprehensive hashing, so each distinct `model_class` and `file_path` gets its own cached instance.
- `DataStorageId` enumerates backend ids: `API`, `CSV`, `DB`, `JSON`, `JSONL`. Only CSV, JSON, and JSONL are implemented.

## CsvDataStorage

- `read()` raises `FileNotFoundError` if the file is missing, and `ValueError` unless the header row equals the dataclass field names in declaration order.
- Values are converted to the declared field types: `str`, `int`, `float`, `bool` (`true/1/yes`, `false/0/no`), Enum by value; empty string becomes `None` for `Optional` fields.
- `read_by_key(key)` matches the field `<model_type>_id`, where `model_type` is the snake_case model class name (`PolicyRule` gives `policy_rule_id`).
- `write()` overwrites the file using each object's `to_dict()`. `write([])` does nothing; it does not truncate the file.

## JsonDataStorage

- `read()` returns `[]` when the file is missing (the ETL treats a missing output file as an empty registry). Nested dataclasses, `list[T]`, Enums, and `Optional` are reconstructed from declared types; absent fields take their defaults; unknown keys are ignored.
- `read_as_dicts()` returns the raw list without deserializing.
- `read_by_key(key)` matches `key_field`, defaulting to `<model_type>_id`.
- `write()` overwrites the file with `to_dict()` output, indented, UTF-8.

## JsonlDataStorage

- Append-only: `write()` and `append()` add one line per record and never overwrite. Objects with `to_dict()` are serialized through it; dicts are written as-is.
- `read()` returns raw dicts, one per non-empty line, or `[]` when the file is missing.
- `read_by_key(key)` returns the first record whose `key_field` matches; `key_field` supports dot paths such as `metadata.trace_id`.

## Adding a backend

1. Add a `<Name>DataStorageConfig(DataStorageConfig)` with the backend's settings.
2. Add `<Name>DataStorage(DataStorage[<Name>DataStorageConfig])` with `_config_data_type` and the three abstract methods.
3. Add its id to `DataStorageId` if new, register it in `DataStorageFactory._TYPES_MAPPING`, and export both classes from `shared/data/__init__.py`.
4. Document it in `../shared/docs/data.md`.
