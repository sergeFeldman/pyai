import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from typing import Any, Dict

from src.config import AppConfig, Configurable

logger = logging.getLogger(__name__)


class DataPreprocessor(Configurable):

    def __init__(self, config: AppConfig):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)
        logger.info(f"Init DataPreprocessor with model: {self.config.model.name}")

    def load_validate(self) -> pd.DataFrame:
        """
        Load and validate the dataset.

        Returns:
            pd.DataFrame: Generated dataset.
        """
        try:
            df = pd.read_csv(self.config.data.in_data_path)

            # Basic validation.
            required_columns = ['string', 'label']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Dataset must contain columns: {required_columns}")

            # Validate labels.
            valid_labels = set(self.config.classes)
            invalid_labels = set(df['label']) - valid_labels
            if invalid_labels:
                raise ValueError(f"Invalid labels found: {invalid_labels}. Expected: {valid_labels}")

            return df

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.config.data.in_data_path}")

    @staticmethod
    def preprocess_string(input_string: str) -> str:
        """
        Basic string cleaning with validation.

        Args:
            input_string (str): Provided string to be cleaned.

        Returns:
            str: Processed string.
        """
        if not isinstance(input_string, str):
            return ""

        # Remove extra whitespace and basic cleaning.
        target_string = ' '.join(input_string.split())
        return target_string.strip()

    def prepare_datasets(self) -> Dict[str, Any]:
        """
        Prepare training and testing datasets.

        Returns:
            Dict: Generated mapping.
        """
        df = self.load_validate()

        # Clean.
        df['clean_string'] = df['string'].apply(self.preprocess_string)

        # Remove empty strings.
        init_count = len(df)
        df = df[df['clean_string'].str.len() > 0]
        remove_count = init_count - len(df)
        if remove_count > 0:
            logger.warning(f"Removed {remove_count} empty strings.")

        # Create label mappings.
        unique_labels = sorted(set(df['label']))
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        id2label = {idx: label for label, idx in label2id.items()}

        logger.info(f"Label mappings: {label2id}")

        # Convert labels to integers using the mapping.
        df['label_id'] = df['label'].map(label2id)

        # Split the data - use the integer labels
        train_strings, test_strings, train_labels, test_labels = train_test_split(
            df['clean_string'].tolist(),
            df['label_id'].tolist(),
            test_size=self.config.data.test_size,
            random_state=self.config.data.random_state,
            stratify=df['label_id']
        )

        logger.info(f"Tokenizing {len(train_strings)} training and {len(test_strings)} test samples")

        # Tokenize.
        train_encodings = self.tokenizer(
            train_strings,
            truncation=True,
            padding=True,
            max_length=self.config.model.max_length,
            return_tensors=None
        )
        test_encodings = self.tokenizer(
            test_strings,
            truncation=True,
            padding=True,
            max_length=self.config.model.max_length,
            return_tensors=None
        )

        # Convert string labels back for dataset info (for logging only)
        train_label_strings = [id2label[label] for label in train_labels]
        test_label_strings = [id2label[label] for label in test_labels]

        dataset_info = {
            'train_samples': len(train_strings),
            'test_samples': len(test_strings),
            'classes': unique_labels,
            'class_distribution': {
                'train': pd.Series(train_label_strings).value_counts().to_dict(),
                'test': pd.Series(test_label_strings).value_counts().to_dict()
            }
        }

        logger.info(f"Dataset prepared: {dataset_info}")
        logger.info(f"Sample train labels (integers): {train_labels[:5]}")
        logger.info(f"Sample test labels (integers): {test_labels[:5]}")

        return {
            'train_encodings': train_encodings,
            'test_encodings': test_encodings,
            'train_labels': train_labels,
            'test_labels': test_labels,
            'label_mappings': {'label2id': label2id, 'id2label': id2label},
            'dataset_info': dataset_info
        }

def create_data_preprocessor_obj(config_path: str = "config/config.yaml") -> DataPreprocessor:
    """
    Public API "Factory" function to create DataPreprocessor instance object.

    Args:
        config_path (str): Path to config file.

    Returns:
        DataPreprocessor: Instantiated DataPreprocessor object.
    """
    return DataPreprocessor(config=AppConfig.read(config_path))
