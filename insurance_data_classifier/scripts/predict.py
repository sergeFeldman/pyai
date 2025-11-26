"""
Predicting orchestration for "Insurance Document Classification" process.
"""

import argparse
import sys
import traceback

from src.doc_predictor import create_predictor_obj
import src.utils as utl


def print_prediction_result(result):
    """
    Print prediction result.
    """
    if result['success']:
        print(f"Predicted: {result['label']} (Confidence: {result['confidence']:.4f})")
    else:
        print(f"Error: {result['error']}")


def main():
    """
    Main prediction function (handles complete prediction pipeline).
    """
    logger = utl.config_logging(__name__)
    logger.info("Starting Insurance Document Classification Prediction")

    parser = argparse.ArgumentParser(description='Classify insurance documents')

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--string', type=str, help='String to classify')
    input_group.add_argument('--file', type=str, help='File containing strings to classify')

    # Config path
    parser.add_argument('--config-path', type=str, default="config/config.yaml",
                        help='Path to config file (default: config/config.yaml)')

    args = parser.parse_args()

    try:
        # Initialize predictor.
        predictor = create_predictor_obj(args.config_path)
        logger.info(f"Available classes: {list(predictor.label_mappings['id2label'].values())}")

        # Handle single string input.
        if args.string:
            logger.info("Processing input string")
            result = predictor.predict(args.string)
            print_prediction_result(result)
            if result['success']:
                logger.info(f"Prediction successful: {result['label']} (confidence: {result['confidence']:.4f})")
            else:
                logger.error(f"Prediction failed: {result['error']}")

        # Handle file input.
        elif args.file:
            logger.info(f"Processing file: {args.file}")
            try:
                with open(args.file, 'r', encoding='utf-8') as f:
                    strings = f.read()
                result = predictor.predict(strings)
                print_prediction_result(result)
                if result['success']:
                    logger.info(f"Prediction successful: {result['label']} (confidence: {result['confidence']:.4f})")
                else:
                    logger.error(f"Prediction failed: {result['error']}")
            except FileNotFoundError:
                logger.error(f"File not found: {args.file}")
                print(f"Error: File not found - {args.file}")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Error reading file: {e}")
                print(f"Error reading file: {e}")
                sys.exit(1)

        logger.info("Prediction completed")

    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        print("Error: Model not found.")
        print("Please train the model first or provide the correct path with --model-path")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected prediction error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()