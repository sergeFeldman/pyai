from abc import ABC
from dataclasses import dataclass, field
import logging
import os
from typing import Dict, List
import yaml

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    in_data_path: str = "./data/in/insurance_docs.csv"
    out_data_path: str = "./data/out/"
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class ModelConfig:
    name: str = "distilbert-base-uncased"
    num_labels: int = 5
    max_length: int = 512


@dataclass
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 2e-5
    out_path: str = "./models/insurance_classifier"


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    classes: List[str] = field(default_factory=lambda: [
        "claim_form", "invoice", "legal_correspondence",
        "medical_report", "police_report", "policy_form",
    ])

    @classmethod
    def from_dict(cls, config_dict: dict) -> "AppConfig":
        """
        Create config from dictionary.

        Args:
            config_dict (dict): Dictionary containing configuration information.

        Returns:
            AppConfig: Application Configuration object.
        """
        # Get training config and ensure learning_rate is float.
        training_dict = config_dict.get('training', {})
        if 'learning_rate' in training_dict:
            training_dict['learning_rate'] = float(training_dict['learning_rate'])

        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**training_dict),
            classes=config_dict.get('classes', [])
        )

    def to_dict(self) -> Dict:
        """
        Convert configuration attributes to dictionary.

        Returns:
            Dict: Constructed dictionary.
        """
        return {
            'data': {
                'in_data_path': self.data.in_data_path,
                'out_data_path': self.data.out_data_path,
                'test_size': self.data.test_size,
                'random_state': self.data.random_state
            },
            'model': {
                'name': self.model.name,
                'num_labels': self.model.num_labels,
                'max_length': self.model.max_length
            },
            'training': {
                'batch_size': self.training.batch_size,
                'epochs': self.training.epochs,
                'learning_rate': self.training.learning_rate,
                'out_path': self.training.out_path
            },
            'classes': self.classes
        }

    @classmethod
    def read(cls, config_path: str = "config/config.yaml") -> "AppConfig":
        """
        Load configuration from YAML file.

        Args:
            config_path (str): Configuration file path.

        Returns:
            AppConfig: Application Configuration object.
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    config_dict = yaml.safe_load(file)

                logger.info(f"Loaded configuration from {config_path}")
                return cls.from_dict(config_dict)
            else:
                # Return default config if file doesn't exist.
                logger.warning(f"Config file {config_path} not found, using defaults")
                return cls()
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            return cls()

    def write(self, config_path: str) -> None:
        """
        Write configuration to YAML file.

        Args:
            config_path (str): Path to configuration file.

        Returns:
            None.
        """
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as file:
            yaml.dump(self.to_dict(), file, default_flow_style=False)
        logger.info(f"Saved configuration to {config_path}")


class Configurable(ABC):
    """
    Abstract base class for all configurable types.
    """
    def __init__(self, config: AppConfig):
        self.config = config