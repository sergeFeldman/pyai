from abc import ABC
import csv
from dataclasses import dataclass, fields
from enum import Enum
import os
from typing import List, Type, TypeVar

T = TypeVar('T', bound='Persistable')


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


class Persistable(ABC):
    """
    Abstract base class for persistable dataclasses.
    """
    # Static/class attribute
    PERSISTENCE_PATH = os.path.join('data', 'in')

    @classmethod
    def default_abs_path(cls, file_type='csv') -> str:
        """
        Class method to get default filename for this class.

        Args:
            file_type (str, optional): Defaults to csv.
        Returns:
            Path: {PERSISTENCE_PATH}/{class_name_lower}.csv
        """
        return os.path.join(cls.PERSISTENCE_PATH,f"{cls.__name__.lower()}.{file_type}")

    def to_dict(self) -> dict:
        """
        Convert instance to dictionary.
        """
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def read(cls: Type[T], instance_type=None) -> List[T]:
        """
        Load collection of instances from CSV file.
        """
        target_instances = []
        filename = cls.default_abs_path()

        with open(filename, 'r', newline='') as file:
            reader = csv.reader(file)
            headers = next(reader)

            for row in reader:
                if len(row) != len(headers):
                    continue

                # Type conversion mapping.
                type_converters = {
                    str: str,
                    int: int,
                    float: float,
                    bool: lambda x: x.lower() == 'true'
                }

                # Convert attributes.
                attributes = [
                    type_converters.get(field.type, str)(value)
                    for field, value in zip(fields(cls), row)
                ]

                target_instances.append(cls(*attributes))

        if instance_type == 'dict':
            target_instances = [instance.to_dict() for instance in target_instances]

        return target_instances

    @classmethod
    def write(cls: Type[T], instances: List[T]):
        """
        Persist collection of instances to CSV file.
        """
        if not instances:
            return

        os.makedirs(cls.PERSISTENCE_PATH, exist_ok=True)
        filename = cls.default_abs_path()

        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)

            # Get field names from dataclass.
            field_names = [field.name for field in fields(cls)]
            writer.writerow(field_names)

            # Write each instance as a row.
            for instance in instances:
                row = [getattr(instance, field_name) for field_name in field_names]
                writer.writerow(row)


@dataclass
class Customer(Persistable):
    customer_id: str
    name: str
    address: str
    phone: str


@dataclass
class Claim(Persistable):
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
