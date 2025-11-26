import logging
import os
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from typing import Dict

from src.config import AppConfig, Configurable

logger = logging.getLogger(__name__)


class DocModelTrainer(Configurable):

    def __init__(self, config: AppConfig):
        super().__init__(config)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model.name,
            num_labels=self.config.model.num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)
        logger.info(f"Init DocModelTrainer with model: {self.config.model.name}")

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute metrics for evaluation.
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}

    def create_dataset(self, encodings: Dict, labels: list) -> torch.utils.data.Dataset:
        """
        Create PyTorch dataset from encodings and labels.
        """

        class DocDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        return DocDataset(encodings, labels)

    def train(self, train_encodings: Dict,
              train_labels: list,
              test_encodings: Dict,
              test_labels: list,
              label_mappings: Dict) -> Trainer:
        """
        Train the model and return trainer instance.
        """
        # Validate input params.
        if not train_encodings or not train_labels:
            raise ValueError("Training data cannot be empty")
        if not test_encodings or not test_labels:
            raise ValueError("Test data cannot be empty")

        logger.info("Creating datasets...")
        train_dataset = self.create_dataset(train_encodings, train_labels)
        eval_dataset = self.create_dataset(test_encodings, test_labels)

        # Ensure output directory exists.
        os.makedirs(self.config.training.out_path, exist_ok=True)

        logger.info("Setting up training params...")
        doc_training_args = TrainingArguments(
            output_dir=self.config.training.out_path,
            num_train_epochs=self.config.training.epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            per_device_eval_batch_size=self.config.training.batch_size,
            learning_rate=self.config.training.learning_rate,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(self.config.training.out_path, 'logs'),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            save_total_limit=2,
            logging_steps=10,
            report_to=None
        )

        logger.info("Creating trainer...")
        doc_trainer = Trainer(
            model=self.model,
            args=doc_training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        logger.info("Starting training...")
        try:
            doc_trainer.train()
        except Exception as e:
            logger.error(f"Error during trainer.train(): {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

        logger.info("Training completed, saving model...")
        doc_trainer.save_model()

        logger.info("Saving tokenizer...")
        self.tokenizer.save_pretrained(self.config.training.out_path)
        logger.info(f"Saved tokenizer to: {self.config.training.out_path}")

        # Save label mappings and config.
        torch.save(label_mappings, os.path.join(self.config.training.out_path, 'label_mappings.pt'))
        self.config.write(os.path.join(self.config.training.out_path, 'config.yaml'))

        # Evaluate final model.
        eval_results = doc_trainer.evaluate()
        logger.info(f"Final evaluation results: {eval_results}")

        return doc_trainer

def create_model_trainer_obj(config_path: str = "config/config.yaml") -> DocModelTrainer:
    """
    Public API "Factory" function to create DocModelTrainer instance object.

    Args:
        config_path (str): Path to config file.

    Returns:
        DocModelTrainer: Instantiated DocModelTrainer object.
    """
    return DocModelTrainer(config=AppConfig.read(config_path))
