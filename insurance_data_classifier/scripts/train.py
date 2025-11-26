"""
Training orchestration for "Insurance Document Classification" process.
"""

import sys

from src.data_preprocessor import create_data_preprocessor_obj
from src.doc_model_trainer import create_model_trainer_obj
import src.utils as utl


def main():
    """
    Main training function (handles complete training pipeline).
    """
    logger = utl.config_logging(__name__)
    logger.info("Starting Insurance Document Classification Training")

    try:
        # Step 1: Preprocess data
        logger.info("Preprocessing data...")
        preprocessor = create_data_preprocessor_obj()
        data = preprocessor.prepare_datasets()

        logger.info(f"Training on {data['dataset_info']['train_samples']} samples")
        logger.info(f"Testing on {data['dataset_info']['test_samples']} samples")
        logger.info(f"Classes: {data['dataset_info']['classes']}")
        logger.info(f"Class distribution - Train: {data['dataset_info']['class_distribution']['train']}")
        logger.info(f"Class distribution - Test: {data['dataset_info']['class_distribution']['test']}")

        # Step 2: Train and persist model.
        logger.info("Training model...")
        trainer = create_model_trainer_obj()
        trainer.train(
            data['train_encodings'],
            data['train_labels'],
            data['test_encodings'],
            data['test_labels'],
            data['label_mappings']
        )

        logger.info("Training completed successfully!")
        logger.info(f"Model and artifacts persisted to: {trainer.config.training.out_path}")

    except ValueError as e:
        logger.error(f"Configuration or data error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Training runtime error (check GPU memory): {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected training error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
