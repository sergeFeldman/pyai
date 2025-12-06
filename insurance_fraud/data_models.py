from dataclasses import dataclass
from enum import Enum
from typing import List


class CLAIM_TYPE(Enum):
    AUTO_COLLISION = 'auto_collision'
    PROPERTY_DAMAGE = 'property_damage'
    THEFT = 'theft'

    @classmethod
    def values(cls) -> List[str]:
        return [member.value for member in CLAIM_TYPE]


class CLAIM_STATUS(Enum):
    CLOSED = 'closed'
    OPEN = 'open'
    UNDER_REVIEW = 'under_review'

    @classmethod
    def values(cls) -> List[str]:
        return [member.value for member in CLAIM_STATUS]


@dataclass
class Customer:
    customer_id: str
    name: str
    address: str
    phone: str

    def to_dict(self) -> dict:
        """
        Convert to dictionary structure.

        Args:
        Returns:
            dict of Customer objects.
        """
        return {
            'customer_id': self.customer_id,
            'name': self.name,
            'phone': self.phone,
            'address': self.address
        }


@dataclass
class Claim:
    claim_id: str
    claim_type: str
    customer_id: str
    amount: float
    date: str
    repair_shop: str
    status: str
    is_fraud: bool = False

    def __repr__(self):
        return (f"Claim(claim_id='{self.claim_id}', "
                f"claim_type='{self.claim_type}', "
                f"customer_id='{self.customer_id}', "
                f"amount={self.amount:.2f}, "
                f"date='{self.date}', "
                f"repair_shop='{self.repair_shop}', "
                f"status='{self.status}', "
                f"is_fraud={self.is_fraud})")

    def to_dict(self) -> dict:
        """
        Convert to dictionary for compatibility with original code.
        """
        return {
            'claim_id': self.claim_id,
            'customer_id': self.customer_id,
            'claim_type': self.claim_type,
            'amount': self.amount,
            'date': self.date,
            'status': self.status,
            'repair_shop': self.repair_shop,
            'is_fraud': self.is_fraud
        }
