"""
Creation of sample insurance data for fraud detection POC.
"""

from datetime import datetime, timedelta
import numpy as np
from typing import List, Tuple

import data_models as dm

CUSTOMER_COUNT = 50
NORMAL_CLAIM_COUNT = 200
FRAUD_CLAIM_COUNT = 20

# Data templates
CUSTOMER_CONFIG = {
    "name_templates": ["Customer_{}", "Client_{}", "PolicyHolder_{}"],
    "phone_templates": ["123-{:03d}", "718-{:03d}", "212-{:03d}"],
    "address_templates": ["{} Main St", "{} Oak Ave", "{} Union Square", "{} Wilson Blvd"]
}

CLAIM_CONFIG = {
    "normal_claims": {
        "types": dm.CLAIM_TYPE.values(),
        "statuses": dm.CLAIM_STATUS.values(),
        "repair_shops": [f"shop_{idx}" for idx in range(1, 11)],
        "amount_range": (1000, 5000),
        "date_range": (0, 365)
    },
    "fraud_claims": {
        "types": ["auto_collision", "theft"],
        "statuses": ["under_review", "open"],
        "repair_shops": ["shop_15", "shop_8"],
        "amount_range": (10000, 15000),
        "date_range": (0, 90)
    },
    "fraud_nodes": {
        "node_1": {
            "customers": ["cust_5", "cust_6", "cust_7"],
            "repair_shop": "shop_15",
            "claim_type": "auto_collision"
        },
        "node_2": {
            "customers": ["cust_25", "cust_26", "cust_27"],
            "repair_shop": "shop_8",
            "claim_type": "theft"
        }
    }
}


def create_sample_data() -> Tuple[List[dict], List[dict]]:
    """
    Create sample insurance data for fraud detection.

    Returns:
        Tuple of (customers, claims) as dictionaries.
    """
    np.random.seed(42)

    print("Creating sample insurance data...")

    customers = _create_customers()
    claims = _create_claims(customers)

    # Convert to dictionaries
    customers_dict = [customer.to_dict() for customer in customers]
    claims_dict = [claim.to_dict() for claim in claims]

    fraud_count = sum(1 for claim in claims if claim.is_fraud)

    print(f"Created {len(customers)} customers and {len(claims)} claims")
    print(f"Normal claims: {len(claims) - fraud_count} Fraud claims: {fraud_count}")

    return customers_dict, claims_dict


def _create_customers() -> List[dm.Customer]:
    """
    Create customer data.
    """
    return [
        dm.Customer(
            customer_id=f'cust_{idx}',
            name=_create_name(idx),
            address=_create_address(),
            phone=_create_phone()
        ) for idx in range(1, CUSTOMER_COUNT + 1)
    ]


def _create_claims(customers: List[dm.Customer]) -> List[dm.Claim]:
    """
    Create claim data with embedded fraud patterns.
    """
    target_claims = []
    customer_ids = [customer.customer_id for customer in customers]

    # Create normal claims.
    target_claims += [
        dm.Claim(
            claim_id=f'claim_{idx}',
            customer_id=np.random.choice(customer_ids),
            claim_type=np.random.choice(CLAIM_CONFIG["normal_claims"]["types"]),
            amount=float(_create_amount(CLAIM_CONFIG["normal_claims"]["amount_range"])),
            date=_create_date(CLAIM_CONFIG["normal_claims"]["date_range"]),
            status=np.random.choice(CLAIM_CONFIG["normal_claims"]["statuses"]),
            repair_shop=np.random.choice(CLAIM_CONFIG["normal_claims"]["repair_shops"]),
            is_fraud=False
        ) for idx in range(1, NORMAL_CLAIM_COUNT + 1)
    ]

    # Create fraud node 1 claims.
    node_1_config = CLAIM_CONFIG["fraud_nodes"]["node_1"]
    target_claims += [
        dm.Claim(
            claim_id=f'claim_{idx}',
            customer_id=np.random.choice(node_1_config["customers"]),
            claim_type=node_1_config["claim_type"],
            amount=float(_create_amount(CLAIM_CONFIG["fraud_claims"]["amount_range"])),
            date=_create_date(CLAIM_CONFIG["fraud_claims"]["date_range"]),
            status=np.random.choice(CLAIM_CONFIG["fraud_claims"]["statuses"]),
            repair_shop=node_1_config["repair_shop"],
            is_fraud=True
        ) for idx in range(NORMAL_CLAIM_COUNT + 1, NORMAL_CLAIM_COUNT + FRAUD_CLAIM_COUNT // 2 + 1)
    ]

    # Create fraud node 2 claims.
    node_2_config = CLAIM_CONFIG["fraud_nodes"]["node_2"]
    target_claims += [
        dm.Claim(
            claim_id=f'claim_{idx}',
            customer_id=np.random.choice(node_2_config["customers"]),
            claim_type=node_2_config["claim_type"],
            amount=float(_create_amount(CLAIM_CONFIG["fraud_claims"]["amount_range"])),
            date=_create_date(CLAIM_CONFIG["fraud_claims"]["date_range"]),
            status=np.random.choice(CLAIM_CONFIG["fraud_claims"]["statuses"]),
            repair_shop=node_2_config["repair_shop"],
            is_fraud=True
        ) for idx in range(NORMAL_CLAIM_COUNT + FRAUD_CLAIM_COUNT // 2 + 1,
                           NORMAL_CLAIM_COUNT + FRAUD_CLAIM_COUNT + 1)
    ]

    return target_claims


def _create_name(idx: int) -> str:
    """
    Create a customer name using configuration templates.
    """
    template = np.random.choice(CUSTOMER_CONFIG["name_templates"])
    return template.format(idx)


def _create_phone() -> str:
    """
    Create a random phone number using configuration templates.
    """
    template = np.random.choice(CUSTOMER_CONFIG["phone_templates"])
    return template.format(np.random.randint(100, 999))


def _create_address() -> str:
    """
    Create a random address using configuration templates.
    """
    template = np.random.choice(CUSTOMER_CONFIG["address_templates"])
    return template.format(np.random.randint(1, 9999))


def _create_amount(amount_range: Tuple[float, float]) -> float:
    """
    Create a random claim amount.
    """
    return max(amount_range[0], np.random.lognormal(
        np.log(amount_range[0]),
        (np.log(amount_range[1]) - np.log(amount_range[0])) / 3
    ))


def _create_date(days_range: Tuple[int, int]) -> str:
    """
    Create a random date within range.
    """
    start_date = datetime(2023, 1, 1)
    random_days = np.random.randint(days_range[0], days_range[1])
    return (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")
