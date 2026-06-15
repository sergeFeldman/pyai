"""ETL pipeline for loading, versioning, and persisting rules."""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import cast

import yaml

from pydantic import BaseModel, ConfigDict

import shared.core as shd_core
import shared.data as shd_data
import rules as rls


class RuleEtlConfig(BaseModel):
    """Configuration for a single domain ETL pipeline run.

    Each entry in etl.yaml produces one RuleEtlConfig.

    Attributes:
        domain: Business domain this run is responsible for, e.g. "claim_appeal".
            Stamped onto every rule processed by this pipeline.
        input_file_path: Path to the raw rules JSON file.
        output_file_path: Path to the versioned output JSON file.
        updated_by: Author recorded in rule metadata on creation and on every version bump.
        key_field: Primary key field name on the rule. Defaults to "id".
    """

    model_config = ConfigDict(extra="forbid")

    domain: str
    input_file_path: str
    output_file_path: str
    updated_by: str
    key_field: str = "id"

    @property
    def input_storage_config(self) -> dict:
        """Storage config dict for the raw input file."""
        return {"model_class": rls.Rule, "key_field": self.key_field, "file_path": self.input_file_path}

    @property
    def output_storage_config(self) -> dict:
        """Storage config dict for the versioned output file."""
        return {"model_class": rls.Rule, "key_field": self.key_field, "file_path": self.output_file_path}


class RuleEtl(shd_core.Configurable[RuleEtlConfig]):
    """Versioning ETL pipeline for a single rule domain.

    Processes one domain per run. Each run:
      1. Loads output_file_path into the registry, if it exists.
         Skipped on first run when no output file exists yet.
      2. Reads raw input rules. For each rule: inserts at version 0 if new,
         or bumps its version if any business field has changed. Unchanged rules
         are skipped.
      3. Writes the full registry (all versions) back to output_file_path.

    Input:  raw JSON - rule fields only.
    Output: structured JSON - all rule versions including domain and metadata.
    """

    _config_data_type = RuleEtlConfig

    def __init__(self, config: RuleEtlConfig):
        """Initialize the pipeline with its input/output storages and an empty registry.

        Args:
            config: Validated ETL configuration for this domain.
        """
        super().__init__(config)
        self._input_storage = \
            cast(shd_data.JsonDataStorage, shd_data.DataStorageFactory().get_obj(
                shd_data.DataStorageId.JSON.value, config.input_storage_config))
        self._output_storage = \
            cast(shd_data.JsonDataStorage, shd_data.DataStorageFactory().get_obj(
                shd_data.DataStorageId.JSON.value, config.output_storage_config))
        self._registry = rls.RuleRegistry()

    def run(self) -> None:
        """Execute the ETL pipeline for the configured domain."""
        updated_by = self.config.updated_by
        domain = self.config.domain

        # Step 1 - seed registry from existing output (empty on first run)
        self._registry.load([
            rls.RuleFactory().from_dict(raw)
            for raw in self._output_storage.read_as_dicts()
        ])

        # Step 2 - read raw input, insert new rules or bump changed ones
        # rule class and its valid fields are resolved once per kind, not per rule.
        kind_cache: dict[str, tuple[type, set[str]]] = {}
        for raw in self._input_storage.read_as_dicts():
            kind = rls.RuleFactory().detect_type(raw)
            if kind not in kind_cache:
                rule_class = rls.RuleFactory().get_class(kind)
                kind_cache[kind] = (rule_class, {f.name for f in dataclasses.fields(rule_class)})
            rule_class, valid = kind_cache[kind]
            fields = {k: v for k, v in raw.items() if k in valid}
            existing = self._registry.get_latest(raw["id"], domain)

            if existing is None:
                self._registry.add(
                    rule_class(**fields, domain=domain, metadata=shd_core.EntityMetadata(created_by=updated_by)))
            else:
                incoming = rule_class(**fields, domain=domain, metadata=copy.deepcopy(existing.metadata))
                if existing.is_changed(incoming):
                    incoming.metadata.bump(updated_by)
                    self._registry.add(incoming)

        # Step 3 - persist all rule versions to the domain output file.
        self._output_storage.write(self._registry.all())


def _load_etl_config() -> dict:
    """Load and return the ETL pipeline configuration from config/etl.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config" / "etl.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    """Run all domain ETL pipelines defined in config/etl.yaml."""
    pipelines = _load_etl_config()
    for domain, cfg in pipelines.items():
        print(f"Running ETL for domain: {domain}")
        config = RuleEtlConfig(
            domain=domain,
            input_file_path=cfg["input_file_path"],
            output_file_path=cfg["output_file_path"],
            updated_by=cfg["updated_by"],
        )
        RuleEtl(config).run()
        print(f"  Done -> {cfg['output_file_path']}")


if __name__ == "__main__":
    main()
