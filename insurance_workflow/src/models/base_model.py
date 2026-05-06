"""Base Pydantic model classes."""

from pydantic import BaseModel, ConfigDict


class WorkflowBaseModel(BaseModel):
    """Base Pydantic model with shared validation configuration."""

    model_config = ConfigDict(extra="forbid")
