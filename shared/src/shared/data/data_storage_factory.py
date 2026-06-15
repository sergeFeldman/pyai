"""Data storage factory."""

from __future__ import annotations

import shared.core as shd_core
from .csv_data_storage import CsvDataStorage
from .json_data_storage import JsonDataStorage


class DataStorageFactory(shd_core.ConfigurableObjectFactory):
    """Singleton factory for creating and caching DataStorage objects.

    Resolves storage backend identifiers (e.g. ``"csv"``, ``"json"``) to concrete
    DataStorage subclasses via the class-level ``_TYPES_MAPPING`` registry.
    """

    _TYPES_MAPPING = {
        "csv": CsvDataStorage,
        "json": JsonDataStorage,
    }

    def __init__(self):
        # Comprehensive hashing required: multiple storage instances share the
        # same storage type id (e.g. "csv") but differ by model_class and file_path.
        # Without it, all CSV storages collapse to the same cache slot.
        super().__init__(comprehensive_hashing=True)
