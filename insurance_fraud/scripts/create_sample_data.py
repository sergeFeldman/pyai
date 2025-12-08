"""
Creation of sample insurance data for fraud detection POC.
"""
from datetime import datetime, timedelta
import numpy as np
from typing import List, Tuple

import src.data_models as dm
import src.utils as utl

# Configuration
RANDOM_SEED = 42
CUSTOMER_COUNT = 50
NORMAL_CLAIM_COUNT = 200
FRAUD_CLAIM_COUNT = 20

# Data templates
CUSTOMER_CONFIG = {
    "name_templates": ["Customer_{}", "Client_{}", "PolicyHolder_{}"],
    "phone_templates": ["123-{:03d}-{:04d}", "718-{:03d}-{:04d}", "212-{:03d}-{:04d}"],
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

logger = utl.config_logging(__name__)


class InsuranceDataGenerator:
    """
    Generate sample insurance data with configurable embedded fraud patterns.
    """

    def __init__(self, random_seed: int = RANDOM_SEED):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.start_date = datetime(2023, 1, 1)

    def generate_customers(self, target_num: int = CUSTOMER_COUNT) -> List[dm.Customer]:
        """
        Generate customer data with shared contact patterns.

        Args:
            target_num (int; default=CUSTOMER_COUNT): Desired number of generated Customer instances.

        Returns:
            List: Generated collection of Customer instances.
        """
        num_shared = max(2, int(target_num * 0.05))
        shared_phone = self._generate_phone_number()
        shared_address = self._generate_address()

        phone_shared_idx = set(np.random.choice(range(1, target_num + 1), size=num_shared, replace=False))
        address_shared_idx = set(np.random.choice(range(1, target_num + 1), size=num_shared, replace=False))

        return [
            dm.Customer(
                customer_id=f'cust_{i}',
                name=self._generate_customer_name(i),
                phone=shared_phone if i in phone_shared_idx else self._generate_phone_number(),
                address=shared_address if i in address_shared_idx else self._generate_address()
            ) for i in range(1, target_num + 1)
        ]

    def generate_claims(self, customers: List[dm.Customer], target_num: int = NORMAL_CLAIM_COUNT,
                        target_fraud_num: int = FRAUD_CLAIM_COUNT) -> List[dm.Claim]:
        """
        Generate claim data with embedded fraud patterns.

        Args:
            customers (List): Provided collection of Customer instances.
            target_num (int; default=NORMAL_CLAIM_COUNT): Desired number of generated normal Claim instances.
            target_fraud_num (int; default=FRAUD_CLAIM_COUNT): Desired number of generated fraudulent Claim instances.

        Returns:
            List: Generated collection of all Claim instances.
        """
        customer_ids = [c.customer_id for c in customers]
        normal_claims = [
            dm.Claim(
                claim_id=f'claim_{i}',
                customer_id=np.random.choice(customer_ids),
                claim_type=np.random.choice(CLAIM_CONFIG["normal_claims"]["types"]),
                amount=self._generate_amount(CLAIM_CONFIG["normal_claims"]["amount_range"]),
                date=self._generate_date(CLAIM_CONFIG["normal_claims"]["date_range"]),
                status=np.random.choice(CLAIM_CONFIG["normal_claims"]["statuses"]),
                repair_shop=np.random.choice(CLAIM_CONFIG["normal_claims"]["repair_shops"]),
                is_fraud=False
            ) for i in range(1, target_num + 1)
        ]

        node_1_config = CLAIM_CONFIG["fraud_nodes"]["node_1"]
        fraud_node_1 = [
            dm.Claim(
                claim_id=f'claim_{i}',
                customer_id=np.random.choice(node_1_config["customers"]),
                claim_type=node_1_config["claim_type"],
                amount=self._generate_amount(CLAIM_CONFIG["fraud_claims"]["amount_range"]),
                date=self._generate_date(CLAIM_CONFIG["fraud_claims"]["date_range"]),
                status=np.random.choice(CLAIM_CONFIG["fraud_claims"]["statuses"]),
                repair_shop=node_1_config["repair_shop"],
                is_fraud=True
            ) for i in range(target_num + 1, target_num + target_fraud_num // 2 + 1)
        ]

        node_2_config = CLAIM_CONFIG["fraud_nodes"]["node_2"]
        fraud_node_2 = [
            dm.Claim(
                claim_id=f'claim_{i}',
                customer_id=np.random.choice(node_2_config["customers"]),
                claim_type=node_2_config["claim_type"],
                amount=self._generate_amount(CLAIM_CONFIG["fraud_claims"]["amount_range"]),
                date=self._generate_date(CLAIM_CONFIG["fraud_claims"]["date_range"]),
                status=np.random.choice(CLAIM_CONFIG["fraud_claims"]["statuses"]),
                repair_shop=node_2_config["repair_shop"],
                is_fraud=True
            ) for i in range(target_num + target_fraud_num // 2 + 1,
                             target_num + target_fraud_num + 1)
        ]

        return normal_claims + fraud_node_1 + fraud_node_2

    def _generate_customer_name(self, idx: int) -> str:
        """
        Generate a customer name.
        """
        template = np.random.choice(CUSTOMER_CONFIG["name_templates"])
        return template.format(idx)

    def _generate_phone_number(self) -> str:
        """
        Generate a phone number.
        """
        template = np.random.choice(CUSTOMER_CONFIG["phone_templates"])
        return template.format(
            np.random.randint(100, 999),
            np.random.randint(1000, 9999)
        )

    def _generate_address(self) -> str:
        """
        Generate an address.
        """
        template = np.random.choice(CUSTOMER_CONFIG["address_templates"])
        return template.format(np.random.randint(1, 9999))

    def _generate_amount(self, amount_range: Tuple[float, float]) -> float:
        """
        Generate a claim amount.
        """
        min_amount, max_amount = amount_range
        amount = max(min_amount, np.random.lognormal(
            np.log(min_amount), (np.log(max_amount) - np.log(min_amount)) / 3))
        return round(amount, 2)

    def _generate_date(self, days_range: Tuple[int, int]) -> str:
        """
        Generate a date within the specified range.
        """
        min_days, max_days = days_range
        days_offset = np.random.randint(min_days, max_days)
        claim_date = self.start_date + timedelta(days=days_offset)
        return claim_date.strftime('%Y-%m-%d')


def create_sample_data(random_seed: int = RANDOM_SEED) -> None:
    """
    Create and save sample insurance data.
    """
    logger.info(f"Starting creation of sample insurance data (seed: {random_seed})...")

    data_generator = InsuranceDataGenerator(random_seed=random_seed)

    customers = data_generator.generate_customers(CUSTOMER_COUNT)
    claims = data_generator.generate_claims(customers, NORMAL_CLAIM_COUNT, FRAUD_CLAIM_COUNT)

    # Persist to CSV.
    dm.Customer.write(customers)
    dm.Claim.write(claims)

    # Summary stats.
    fraud_count = sum(1 for claim in claims if claim.is_fraud)

    logger.info(f"Generated {len(customers)} customers")
    logger.info(f"Generated {len(claims)} claims")
    logger.info(f"  - Normal claims: {len(claims) - fraud_count}")
    logger.info(f"  - Fraud claims: {fraud_count}")
    logger.info(f"Files saved to {dm.Customer.PERSISTENCE_PATH} directory:")
    logger.info(f"  - {dm.Customer.default_abs_path()}")
    logger.info(f"  - {dm.Claim.default_abs_path()}")


if __name__ == '__main__':
    create_sample_data()
