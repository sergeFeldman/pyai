import logging
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Any, Dict, List

from src.config import AppConfig, Configurable

logger = logging.getLogger(__name__)


class DocPredictor(Configurable):

    def __init__(self, config: AppConfig):
        super().__init__(config)

        # Use model path from config attribute.
        self.model_path = self.config.training.out_path

        # Load model components.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

        # Explicit CPU assignment and evaluation mode.
        self.model.to(torch.device("cpu"))
        self.model.eval()

        # Load label mappings.
        label_mappings_path = os.path.join(self.model_path, 'label_mappings.pt')
        if os.path.exists(label_mappings_path):
            self.label_mappings = torch.load(label_mappings_path)
        else:
            self.label_mappings = {
                'id2label': {idx: label for idx, label in enumerate(self.config.classes)},
                'label2id': {label: idx for idx, label in enumerate(self.config.classes)}
            }

        # Create pipeline with CPU device.
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1,
            return_all_scores=False
        )

        logger.info(f"Init DocPredictor with model: {self.config.model.name}")
        logger.debug(f"Available classes: {list(self.label_mappings['label2id'].keys())}")

    def predict(self, input_string: str) -> Dict[str, Any]:
        """
        Predict class for provided string.

        Args:
            input_string (str): Provided string.

        Returns:
            Dict: Generated mapping.
        """
        # Input validation (our responsibility)
        if not input_string or not isinstance(input_string, str):
            return {
                'label': 'error',
                'orig_label': 'LABEL_ERROR',
                'confidence': 0.0,
                'error': "Invalid input: provided string must be a non-empty string",
                'success': False
            }

        # Let any model/inference errors propagate to caller.
        result = self.classifier(input_string)[0]
        return self._process_prediction_result(result)

    def predict_batch(self, input_strings: List[str]) -> List[Dict[str, Any]]:
        """
        Predict classes for multiple strings efficiently using batch processing.

        Args:
            input_strings (List): Provided collection of strings.

        Returns:
            List: Generated collection.

        """
        if not input_strings:
            return []

        # Input validation.
        valid_strings = []
        invalid_indices = []

        for idx, input_string in enumerate(input_strings):
            if input_string and isinstance(input_string, str):
                valid_strings.append(input_string)
            else:
                invalid_indices.append(idx)

        if not valid_strings:
            return [{
                'label': 'error',
                'orig_label': 'LABEL_ERROR',
                'confidence': 0.0,
                'error': "Invalid input: empty string list",
                'success': False
            } for _ in input_strings]

        # Let batch processing errors propagate.
        results = self.classifier(valid_strings)
        predictions = [self._process_prediction_result(result) for result in results]

        # Handle invalid inputs.
        if invalid_indices:
            final_predictions = []
            valid_idx = 0
            for idx in range(len(input_strings)):
                if idx in invalid_indices:
                    final_predictions.append({
                        'label': 'error',
                        'orig_label': 'LABEL_ERROR',
                        'confidence': 0.0,
                        'error': "Invalid input string",
                        'success': False
                    })
                else:
                    final_predictions.append(predictions[valid_idx])
                    valid_idx += 1
            return final_predictions

        return predictions

    def _extract_label_id(self, label_string: str) -> int:
        """
        Extract label ID from classifier output string.
        """
        try:
            # Handle different label formats: 'LABEL_1', '1', etc.
            if label_string.startswith('LABEL_'):
                return int(label_string.split('_')[-1])
            else:
                return int(label_string)
        except (ValueError, IndexError) as e:
            logger.error(f"Failed to extract label ID from '{label_string}': {e}")
            raise ValueError(f"Invalid label format: {label_string}")

    def _process_prediction_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal method to process a single prediction result.
        """
        try:
            # Extract label ID and convert to human-readable label.
            predicted_id = self._extract_label_id(result['label'])
            predicted_label = self.label_mappings['id2label'][predicted_id]

            return {
                'label': predicted_label,
                'orig_label': result['label'],
                'confidence': float(result['score']),
                'success': True
            }

        except (KeyError, ValueError, IndexError) as e:
            logger.warning(f"Error processing prediction result: {e}, result: {result}")
            return {
                'label': 'error',
                'orig_label': 'LABEL_ERROR',
                'confidence': 0.0,
                'error': f"Result processing error: {e}",
                'success': False
            }


def create_predictor_obj(config_path: str = "config/config.yaml") -> DocPredictor:
    """
    Public API "Factory" function to create DocPredictor instance object.

    Args:
        config_path (str): Path to config file.

    Returns:
        DocPredictor: Instantiated DocPredictor object.
    """
    # Read config from specified path.
    config = AppConfig.read(config_path)

    # Validate model path exists
    if not os.path.exists(config.training.out_path):
        raise FileNotFoundError(f"Model path not found: {config.training.out_path}")

    return DocPredictor(config=config)
