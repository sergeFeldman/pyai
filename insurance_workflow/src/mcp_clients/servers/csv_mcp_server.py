"""CSV-backed MCP server exposing claim, customer context, and policy rule tools.

Run from the project root with:
    PYTHONPATH=src python src/mcp_clients/servers/csv_mcp_server.py
"""

from mcp.server.fastmcp import FastMCP

import data

mcp = FastMCP("csv-server")


@mcp.tool()
def get_claim(claim_id: str) -> dict:
    """Retrieve a claim record by claim ID.

    Args:
        claim_id: Unique claim identifier, e.g. 'claim_42'.

    Returns:
        Claim record as a dictionary, or empty dict if not found.
    """
    storage = data.DataStorageFactory().get_obj(data.DataStorageId.CSV.value,
                                                {"model_type": data.DataModelType.CLAIM,
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
    storage = data.DataStorageFactory().get_obj(data.DataStorageId.CSV.value,
                                                {"model_type": data.DataModelType.CUSTOMER,
                                                 "file_path": "data/in/customer_context.csv"})
    context = storage.read_by_key(customer_id)
    return context.to_dict() if context else {}


@mcp.tool()
def get_policy_rule(policy_rule_id: str) -> dict:
    """Retrieve a policy rule by its primary key.

    Args:
        policy_rule_id: Unique policy rule identifier, e.g. 'rule_1'.

    Returns:
        Policy rule as a dictionary, or empty dict if not found.
    """
    storage = data.DataStorageFactory().get_obj(data.DataStorageId.CSV.value,
                                                {"model_type": data.DataModelType.POLICY_RULE,
                                                 "file_path": "data/in/policy_rules.csv"})
    rule = storage.read_by_key(policy_rule_id)
    return rule.to_dict() if rule else {}


@mcp.tool()
def get_policy_rule_by_filter(claim_type: str, attribute: str, value: str) -> dict:
    """Retrieve the applicable policy rule for a given claim type, attribute, and value.

    Args:
        claim_type: Claim type, e.g. 'auto_collision', 'theft', 'property_damage'.
        attribute: Claim attribute being explained, e.g. 'status', 'is_fraud'.
        value: Attribute value, e.g. 'denied', 'true'.

    Returns:
        Matching policy rule as a dictionary, or empty dict if not found.
    """
    storage = data.DataStorageFactory().get_obj(data.DataStorageId.CSV.value,
                                                {"model_type": data.DataModelType.POLICY_RULE,
                                                 "file_path": "data/in/policy_rules.csv"})
    for rule in storage.read():
        if rule.claim_type == claim_type and rule.attribute == attribute and rule.value == value:
            return rule.to_dict()
    return {}


if __name__ == "__main__":
    mcp.run()
