"""Application dependency wiring related functions."""

import os
from typing import Type
import yaml

import shared.data as shd_data
import handlers as hdl
import mcp_clients as mcp
import models as mdl
import services as svc
import workflow as wfl

_MODEL_CLASS_MAPPING: dict[str, type] = {
    "claim": mdl.Claim,
    "customer": mdl.Customer,
    "policy_rule": mdl.PolicyRule,
    "claim_appeal_rule": mdl.ClaimAppealRule,
}


def _load_config(config_file: str, name: str) -> dict:
    """Load a named entry from a YAML config file in the config/ directory.

    Args:
        config_file: YAML filename, e.g. 'agents.yaml' or 'storage.yaml'.
        name: Top-level key to retrieve from the file.

    Returns:
        dict: Configuration values for the requested entry.

    Raises:
        FileNotFoundError: Raised if the config file does not exist.
        KeyError: Raised if the requested key is not present in the file.
    """
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", config_file)
    with open(os.path.normpath(config_path)) as f:
        return yaml.safe_load(f)[name]


def _build_mcp_client_config(key: str, client_config_type: Type):
    """Build an MCP client config from a storage.yaml entry.

    Reads the named entry from config/storage.yaml, resolves storage_type to a
    DataStorageId and model_type to a concrete model class via _MODEL_CLASS_MAPPING.

    Args:
        key: Entry name in storage.yaml, e.g. 'claim', 'customer', 'policy_rule'.
        client_config_type: MCP client config class to instantiate.

    Returns:
        mcp.MpcClientConfig: Fully constructed MCP client config instance.
    """
    config = _load_config("storage.yaml", key)
    model_type = config["model_type"]
    return client_config_type(
        data_storage_id=shd_data.DataStorageId(config["storage_type"]),
        data_storage_config={
            "model_class": _MODEL_CLASS_MAPPING[model_type],
            "file_path": config["file_path"],
        },
    )


def get_claim_agent_config() -> dict:
    """Build the claim-agent configuration from config/storage.yaml.

    Returns:
        dict: Claim-agent configuration keyed by 'claim_mcp_client_config'.
    """
    return {"claim_mcp_client_config": _build_mcp_client_config("claim", mcp.ClaimMcpClientConfig)}


def get_customer_agent_config() -> dict:
    """Build the customer-agent configuration from config/storage.yaml.

    Returns:
        dict: Customer-agent configuration keyed by 'customer_mcp_client_config'.
    """
    return {"customer_mcp_client_config": _build_mcp_client_config("customer", mcp.CustomerMcpClientConfig)}


def get_policy_rule_agent_config() -> dict:
    """Build the policy rule agent configuration from config/storage.yaml.

    Returns:
        dict: Policy rule agent configuration keyed by 'policy_rule_mcp_client_config'.
    """
    return {"policy_rule_mcp_client_config": _build_mcp_client_config("policy_rule", mcp.PolicyRuleMcpClientConfig)}


def get_claim_appeal_agent_config() -> dict:
    """Build the claim appeal agent configuration from config/storage.yaml.

    Returns:
        dict: Claim appeal agent configuration keyed by 'claim_appeal_rule_mcp_client_config'.
    """
    return {"claim_appeal_rule_mcp_client_config":
            _build_mcp_client_config("claim_appeal_rule", mcp.ClaimAppealRuleMcpClientConfig)}


def get_claim_explanation_agent_config() -> dict:
    """Build the claim explanation agent configuration from config/agents.yaml.

    Reads the 'claim_explanation' entry containing llm_provider, model,
    and prompt_name.

    Returns:
        dict: Claim explanation agent configuration.
    """
    return _load_config("agents.yaml", "claim_explanation")


_AGENT_CONFIGS = {
    "claim": get_claim_agent_config(),
    "claim_appeal": get_claim_appeal_agent_config(),
    "claim_explanation": get_claim_explanation_agent_config(),
    "customer": get_customer_agent_config(),
    "policy_rule": get_policy_rule_agent_config(),
}


def get_trace_service() -> svc.TraceService:
    """Create the trace service dependency.

    Returns:
        svc.TraceService: Trace service instance.
    """
    return svc.TraceService()


def get_audit_service() -> svc.AuditService:
    """Create the audit service dependency.

    Returns:
        svc.AuditService: Audit service instance.
    """
    return svc.AuditService()


def get_workflow_orchestrator() -> wfl.WorkflowOrchestrator:
    """Create the workflow orchestrator dependency.

    Builds the full agent_configs dict so that routes never need to
    inject or pass agent configuration.

    Returns:
        wfl.WorkflowOrchestrator: Orchestrator instance wired with the trace service,
            audit service, and all agent configs.
    """
    return wfl.WorkflowOrchestrator(
        trace_service=get_trace_service(),
        audit_service=get_audit_service(),
        agent_configs=_AGENT_CONFIGS,
    )


def get_request_handler() -> hdl.RequestHandler:
    """Create the request handler dependency.

    Returns:
        hdl.RequestHandler: Request handler instance wired with the workflow orchestrator.
    """
    return hdl.RequestHandler(workflow_orchestrator=get_workflow_orchestrator())
