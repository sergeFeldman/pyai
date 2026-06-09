"""Data storage factory related classes."""

from __future__ import annotations

import core
import data


class DataStorageFactory(core.ConfigurableObjectFactory):
    """Singleton factory class for creating and caching DataStorage objects.

    This factory inherits from core.ConfigurableObjectFactory for data storage
    implementations. It resolves storage backend identifiers through the
    class-level `_TYPES_MAPPING` registry and returns configured storage
    instances such as CSV, database, or API-backed storages.
    """

    _TYPES_MAPPING = {
        data.DataStorageId.CSV.value: data.CsvDataStorage,
    }

    def __init__(self):
        # Comprehensive hashing required: multiple storage backends share the same
        # storage type id (e.g. "csv") but differ by file_path and model_type.
        # Without it, all CSV storages collapse to the same cache slot.
        super().__init__(comprehensive_hashing=True)
