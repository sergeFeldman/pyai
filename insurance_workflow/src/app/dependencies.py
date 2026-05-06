"""Application dependency wiring related functions."""

import os
import yaml

import data
import handlers as hdl
import mcp_clients as mcp
import services as svc
import workflow as wfl


def _load_agent_config(name: str) -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "agents.yaml")
    with open(os.path.normpath(config_path)) as f:
        return yaml.safe_load(f)[name]


def get_claim_agent_config() -> dict:
    """Build the claim-agent configuration for the current app environment."""
    return {
        "claim_mcp_client_config": mcp.ClaimMcpClientConfig(
            data_storage_id=data.DataStorageId.CSV,
            data_storage_config={
                "model_type": data.DataModelType.CLAIM,
                "file_path": "data/in/claim.csv",
            },
        ),
    }


def get_customer_agent_config() -> dict:
    """Build the customer-agent configuration for the current app environment."""
    return {
        "customer_mcp_client_config": mcp.CustomerMcpClientConfig(
            data_storage_id=data.DataStorageId.CSV,
            data_storage_config={
                "model_type": data.DataModelType.CUSTOMER,
                "file_path": "data/in/customer_context.csv",
            },
        ),
    }


def get_policy_rule_agent_config() -> dict:
    """Build the policy rule agent configuration for the current app environment."""
    return {
        "policy_rule_mcp_client_config": mcp.PolicyRuleMcpClientConfig(
            data_storage_id=data.DataStorageId.CSV,
            data_storage_config={
                "model_type": data.DataModelType.POLICY_RULE,
                "file_path": "data/in/policy_rules.csv",
            },
        ),
    }


def get_claim_explanation_agent_config() -> dict:
    """Build the claim explanation agent configuration for the current app environment."""
    return _load_agent_config("claim_explanation")


def get_trace_service() -> svc.TraceService:
    """Create the trace service dependency."""
    return svc.TraceService()


def get_workflow_orchestrator() -> wfl.WorkflowOrchestrator:
    """Create the workflow orchestrator dependency."""
    return wfl.WorkflowOrchestrator(
        trace_service=get_trace_service(),
    )


def get_request_handler() -> hdl.RequestHandler:
    """Create the request handler dependency."""
    return hdl.RequestHandler(
        workflow_orchestrator=get_workflow_orchestrator(),
    )
