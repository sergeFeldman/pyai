from abc import ABC
from dataclasses import dataclass
import logging
import os
from typing import Dict
import yaml

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """
    Configuration for data generation and management.
    """
    customer_count: int
    normal_claim_count: int
    fraud_claim_count: int
    data_path: str
    embed_path: str
    visual_path: str


@dataclass
class KgConfig:
    """
    Configuration for knowledge graph construction.
    """
    enable_phone_relations: bool
    enable_address_relations: bool
    relation_weights: Dict[str, float]


@dataclass
class KgEmbeddingConfig:
    """
    Configuration for knowledge graph embeddings.
    """
    model_name: str
    hidden_dim: int
    batch_size: int
    gamma: float
    learning_rate: float
    max_step: int
    neg_sample_size: int
    regularization_coef: float


@dataclass
class ModelConfig:
    """
    Configuration for fraud prediction model.
    """
    n_estimators: int
    max_depth: int
    test_size: float
    random_state: int


@dataclass
class AppConfig:
    """Main application configuration."""
    data: DataConfig
    kg: KgConfig
    kg_embedding: KgEmbeddingConfig
    model: ModelConfig

    @classmethod
    def from_dict(cls, config_dict: dict) -> "AppConfig":
        """
        Create config from dictionary. YAML is the single source of truth.
        """
        # Extract nested configs
        data_dict = config_dict['data']
        kg_dict = config_dict['kg']
        kg_embedding_dict = config_dict['kg_embedding']
        model_dict = config_dict['model']

        return cls(
            data=DataConfig(
                customer_count=data_dict['customer_count'],
                normal_claim_count=data_dict['normal_claim_count'],
                fraud_claim_count=data_dict['fraud_claim_count'],
                data_path=data_dict['data_path'],
                embed_path=data_dict['embed_path'],
                visual_path=data_dict['visual_path']
            ),
            kg=KgConfig(
                enable_phone_relations=kg_dict['enable_phone_relations'],
                enable_address_relations=kg_dict['enable_address_relations'],
                relation_weights=kg_dict['relation_weights']
            ),
            kg_embedding=KgEmbeddingConfig(
                model_name=kg_embedding_dict['model_name'],
                hidden_dim=kg_embedding_dict['hidden_dim'],
                batch_size=kg_embedding_dict['batch_size'],
                gamma=kg_embedding_dict['gamma'],
                learning_rate=kg_embedding_dict['learning_rate'],
                max_step=kg_embedding_dict['max_step'],
                neg_sample_size=kg_embedding_dict['neg_sample_size'],
                regularization_coef=kg_embedding_dict['regularization_coef']
            ),
            model=ModelConfig(
                n_estimators=model_dict['n_estimators'],
                max_depth=model_dict['max_depth'],
                test_size=model_dict['test_size'],
                random_state=model_dict['random_state']
            )
        )

    def to_dict(self) -> Dict:
        """
        Convert configuration attributes to dictionary.
        """
        return {
            'data': {
                'customer_count': self.data.customer_count,
                'normal_claim_count': self.data.normal_claim_count,
                'fraud_claim_count': self.data.fraud_claim_count,
                'data_path': self.data.data_path,
                'embed_path': self.data.embed_path,
                'visual_path': self.data.visual_path
            },
            'kg': {
                'enable_phone_relations': self.kg.enable_phone_relations,
                'enable_address_relations': self.kg.enable_address_relations,
                'relation_weights': self.kg.relation_weights
            },
            'kg_embedding': {
                'model_name': self.kg_embedding.model_name,
                'hidden_dim': self.kg_embedding.hidden_dim,
                'batch_size': self.kg_embedding.batch_size,
                'gamma': self.kg_embedding.gamma,
                'learning_rate': self.kg_embedding.learning_rate,
                'max_step': self.kg_embedding.max_step,
                'neg_sample_size': self.kg_embedding.neg_sample_size,
                'regularization_coef': self.kg_embedding.regularization_coef
            },
            'model': {
                'n_estimators': self.model.n_estimators,
                'max_depth': self.model.max_depth,
                'test_size': self.model.test_size,
                'random_state': self.model.random_state
            }
        }

    @classmethod
    def read(cls, config_path: str = "config/config.yaml") -> "AppConfig":
        """
        Load configuration from YAML file.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please create the config file with required settings."
            )

        try:
            with open(config_path, 'r') as file:
                config_dict = yaml.safe_load(file)

            if config_dict is None:
                raise ValueError("Configuration file is empty")

            logger.info(f"Loaded configuration from {config_path}")
            return cls.from_dict(config_dict)

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
        except KeyError as e:
            raise ValueError(f"Missing required configuration key: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")

    def write(self, config_path: str = "config/config.yaml") -> None:
        """
        Write current configuration to YAML file.
        """
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as file:
            yaml.dump(self.to_dict(), file, default_flow_style=False, indent=2)
        logger.info(f"Saved configuration to {config_path}")


class Configurable(ABC):
    """
    Abstract base class for all configurable types.
    """
    def __init__(self, config: AppConfig):
        self.config = config
