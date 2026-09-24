"""CSV-backed MCP server exposing claim, customer context, and policy rule tools.

Run from the project root with:
    PYTHONPATH=src python src/mcp_clients/servers/csv_mcp_server.py
"""

from mcp.server.fastmcp import FastMCP

import shared.data as shd_data
import models as mdl
import rules as rls
from mcp_clients.policy_rule_registry_client import PolicyRuleRegistryClient

mcp = FastMCP("csv-server")

# Load policy rules into the registry for this subprocess.
rls.RuleRegistry.load_from("data/out/policy_rules.json")
_policy_client = PolicyRuleRegistryClient()


@mcp.tool()
def get_claim(claim_id: str) -> dict:
    """Retrieve a claim record by claim ID.

    Args:
        claim_id: Unique claim identifier, e.g. 'claim_42'.

    Returns:
        Claim record as a dictionary, or empty dict if not found.
    """
    storage = shd_data.DataStorageFactory().get_obj(shd_data.DataStorageId.CSV.value,
                                                {"model_class": mdl.Claim,
                                                 "file_path": "data/in/claim.csv"})
    claim = storage.read_by_key(claim_id)
    return claim.to_dict() if claim else {}


@mcp.tool()
def get_customer(customer_id: str) -> dict:
    """Retrieve customer relationship context by customer ID.

    Args:
        customer_id: Unique customer identifier, e.g. 'cust_1'.

    Returns:
        Customer context record as a dictionary, or empty dict if not found.
    """
    storage = shd_data.DataStorageFactory().get_obj(shd_data.DataStorageId.CSV.value,
                                                {"model_class": mdl.Customer,
                                                 "file_path": "data/in/customer_context.csv"})
    context = storage.read_by_key(customer_id)
    return context.to_dict() if context else {}


@mcp.tool()
def get_policy_rule(policy_rule_id: str) -> dict:
    """Retrieve a policy rule by its id from the rule registry.

    Args:
        policy_rule_id: Rule identifier, e.g. 'pr_ac_fraud'.

    Returns:
        Policy rule fields (denial_basis, next_steps, policy_section), or empty dict if not found.
    """
    rule = rls.RuleRegistry().get_latest(policy_rule_id, "policy")
    if rule is None or not isinstance(rule, rls.LookupRule):
        return {}
    return {
        "policy_rule_id": rule.id,
        **rule.match_keys,
        **rule.output_values,
    }


@mcp.tool()
def get_policy_rule_by_filter(claim_type: str, attribute: str, value: str) -> dict:
    """Retrieve the applicable policy rule for a given claim type, attribute, and value.

    Args:
        claim_type: Claim type, e.g. 'auto_collision', 'theft', 'property_damage'.
        attribute: Claim attribute being explained, e.g. 'status', 'is_fraud'.
        value: Attribute value, e.g. 'denied', 'true'.

    Returns:
        Matching policy rule fields (denial_basis, next_steps, policy_section), or empty dict.
    """
    context = {"claim_type": claim_type, "attribute": attribute, "value": value}
    rule = _policy_client.find(context)
    if rule is None:
        return {}
    return {
        "policy_rule_id": rule.id,
        "claim_type": claim_type,
        "attribute": attribute,
        "value": value,
        **rule.output_values,
    }


if __name__ == "__main__":
    mcp.run()
