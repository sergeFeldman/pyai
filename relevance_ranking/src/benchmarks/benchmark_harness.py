import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from datetime import datetime, timezone
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
import os
from scipy import stats

import operations as rr_oper
from ranking import run_ranking_async, DEFAULT_OPERATIONS_MAPPING

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARA_DATA_IN_PATH = os.path.join(BASE_DIR, "data", "parbenchmark", "para-nmt.txt")
STS_DATA_IN_PATH = os.path.join(BASE_DIR, "data", "stsbenchmark", "sts-dev.csv")
DATA_OUT_PATH = os.path.join(BASE_DIR, "data", "out")
LOG_DIR = os.path.join(BASE_DIR, "data", "logs")

OPERATIONS = (rr_oper.SemanticSimilarityOperation,
              rr_oper.FuzzySimilarityOperation,
              rr_oper.EntityOverlapSimilarityOperation)

DATASET_CHOICE = "para"  # Change to "sts" or "para"

class BenchmarkHarness:
    def __init__(self, dataset_id: str, log: logging.Logger):
        self.log = log
        self._dataset_id = dataset_id
        self._score_scale = "0-5" if self._dataset_id == 'sts' else "0-1"
        self.results = []

    @property
    def dataset_id(self):
        return self._dataset_id

    @property
    def score_scale(self):
        return self._score_scale

    def read_data(self):
        if self._dataset_id == "sts":
            target_cases = self.read_sts_data(STS_DATA_IN_PATH)
        else:  # "para"
            target_cases = self.read_paranmt_data(PARA_DATA_IN_PATH)
        return target_cases

    def read_sts_data(self, path: str) -> List[Dict[str, Any]]:
        """
        Load STS related CSV data with correct columns format.
        Format: genre, filename, year, id, score, sentence1, sentence2
        """
        df = pd.read_csv(path, sep='\t', header=None,
                         names=['genre', 'filename', 'year', 'id', 'score', 'sentence1', 'sentence2'],
                         quoting=3, on_bad_lines='warn')

        test_cases = []
        for idx, row in df.iterrows():
            test_case = {
                "row_id": idx,
                "pair_id": row['id'],
                "genre": row['genre'],
                "filename": row['filename'],
                "year": row['year'],
                "human_score_0_to_5": float(row['score']),  # Human score 0-5 scale
                "sentence1": row['sentence1'],
                "sentence2": row['sentence2'],
                "metadata": {
                    "dataset": "sts-dev",
                    "load_timestamp":datetime.now(timezone.utc).isoformat(),
                    "total_rows": len(df)
                }
            }
            test_cases.append(test_case)

        self.log.info(f"Loaded {len(test_cases)} test cases from {path}")
        return test_cases

    def read_paranmt_data(self, path: str) -> List[Dict[str, Any]]:
        """
        Load ParaNMT dataset (tab-separated).
        Format: score (0-1), sentence1, sentence2
        """
        df = pd.read_csv(path, sep='\t', header=None,
                         names=['sentence1', 'sentence2', 'score_0_to_1'],
                         quoting=3, on_bad_lines='warn')

        test_cases = []
        for idx, row in df.iterrows():
            test_case = {
                "row_id": idx,
                "pair_id": f"paranmt_{idx}",
                "dataset": "ParaNMT",
                "human_score_0_to_1": float(row['score_0_to_1']),  # Keep as 0-1
                "sentence1": row['sentence1'],
                "sentence2": row['sentence2'],
                "metadata": {
                    "dataset": "ParaNMT",
                    "load_timestamp": datetime.now(timezone.utc).isoformat(),
                    "total_rows": len(df)
                }
            }
            test_cases.append(test_case)

        self.log.info(f"Loaded {len(test_cases)} ParaNMT test cases from {path}")
        self.log.info(f"Score range: {df['score_0_to_1'].min():.3f} to {df['score_0_to_1'].max():.3f}")
        return test_cases

    async def run_ranking_on_cases(self, test_cases: List[Dict[str, Any]],
                                   lang_id: str = "en",
                                   operations: Tuple = DEFAULT_OPERATIONS_MAPPING,
                                   batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Run ranking on all test cases.

        Returns updated test cases with ranking results.
        """
        self.log.info(f"Starting ranking on {len(test_cases)} cases...")

        for i, test_case in enumerate(test_cases):
            try:
                # Run your ranking
                ranking_result = await run_ranking_async(self.log, test_case['sentence1'], test_case['sentence2'],
                                                         lang_id, operations)

                # Extract DOCUMENT-LEVEL score only.
                doc_score_0_to_1 = ranking_result['rank_score']['doc_level']['0']['0']
                if self._dataset_id == "sts":
                    human_score_0_to_1 = test_case['human_score_0_to_5'] / 5.0
                else:
                    human_score_0_to_1 = test_case['human_score_0_to_1']

                # Calculate errors
                error_0_to_1 = abs(human_score_0_to_1 - doc_score_0_to_1)
                error_0_to_5 = error_0_to_1 * 5.0

                # Get tnk_level scores - renamed to tkn_scores_0_to_1
                tkn_scores_0_to_1 = ranking_result['rank_score'].get('tkn_level', {})

                # Update test case with results
                test_case.update({
                    "human_score_0_to_1": float(human_score_0_to_1),
                    "rank_score_0_to_1": float(doc_score_0_to_1), "rank_score_0_to_5": float(doc_score_0_to_1 * 5.0),
                    "tkn_scores_0_to_1": tkn_scores_0_to_1,
                    "error_0_to_1": float(error_0_to_1), "error_0_to_5": float(error_0_to_5),
                    "within_0.5_stars": error_0_to_5 <= 0.5, "within_1_star": error_0_to_5 <= 1.0,
                    "ranking_timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": False
                })

                # Log progress
                if (i + 1) % batch_size == 0:
                    self.log.info(f"Processed {i + 1}/{len(test_cases)} cases")

            except Exception as e:
                self.log.error(f"Error processing row {test_case['row_id']}: {str(e)}")
                test_case.update({
                    "rank_score_0_to_5": None, "rank_score_0_to_1": None, "tkn_scores_0_to_1": None,
                    "error_0_to_5": None, "error_0_to_1": None,
                    "within_0.5_stars": None, "within_1_star": None,
                    "ranking_timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": True, "error_message": str(e)
                })

        self.results = test_cases
        return test_cases

    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics for document-level scores only."""
        valid_results = [r for r in results if not r.get('error', False)]

        if not valid_results:
            return {"error": "No valid results"}

        human_scores = [r['human_score_0_to_1'] for r in valid_results]
        rank_scores = [r['rank_score_0_to_1'] for r in valid_results]
        errors = [r['error_0_to_5'] for r in valid_results]

        metrics = {
            "dataset": self._dataset_id,
            "score_scale": self._score_scale,
            "summary": {
                "total_cases": len(results),
                "successful_cases": len(valid_results),
                "failed_cases": len(results) - len(valid_results),
                "success_rate": (len(valid_results) / len(results)) * 100
            },
            "correlation": {
                "spearman": float(stats.spearmanr(human_scores, rank_scores)[0]),
                "pearson": float(stats.pearsonr(human_scores, rank_scores)[0])
            },
            "error_metrics": {
                "mean_absolute_error": float(np.mean(errors)),
                "median_absolute_error": float(np.median(errors)),
                "root_mean_squared_error": float(np.sqrt(np.mean([e ** 2 for e in errors]))),
                "max_absolute_error": float(np.max(errors)),
                "std_absolute_error": float(np.std(errors))
            },
            "accuracy_metrics": {
                "within_0.5_stars_percentage": (np.mean([r['within_0.5_stars'] for r in valid_results]) * 100),
                "within_1_star_percentage": (np.mean([r['within_1_star'] for r in valid_results]) * 100)
            },
            "score_distribution": {
                "human_mean": float(np.mean(human_scores)),
                "human_std": float(np.std(human_scores)),
                "rank_mean": float(np.mean(rank_scores)),
                "rank_std": float(np.std(rank_scores)),
                "human_min": float(np.min(human_scores)),
                "human_max": float(np.max(human_scores)),
                "rank_min": float(np.min(rank_scores)),
                "rank_max": float(np.max(rank_scores))
            }
        }

        return metrics

    def write_results(self,
                      results: List[Dict[str, Any]],
                      output_path: str = DATA_OUT_PATH) -> str:
        """
        Save results to CSV file in the same format as input plus generated scores.

        Returns path to saved CSV.
        """
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Create DataFrame
        data = []
        for r in results:
            # Base fields common to both datasets
            row = {
                "row_id": r['row_id'],
                "pair_id": r['pair_id'],
                "sentence1": r['sentence1'],
                "sentence2": r['sentence2'],
                "human_score_0_to_1": r.get('human_score_0_to_1'),
                "rank_score_0_to_1": r.get('rank_score_0_to_1'),
                "rank_score_0_to_5": r.get('rank_score_0_to_5'),
                "tkn_scores_0_to_1": r.get('tkn_scores_0_to_1'),
                "error_0_to_1": r.get('error_0_to_1'),
                "error_0_to_5": r.get('error_0_to_5'),
                "within_0.5_stars": r.get('within_0.5_stars'),
                "within_1_star": r.get('within_1_star'),
                "error": r.get('error', False),
                "error_message": r.get('error_message', ''),
                "ranking_timestamp": r.get('ranking_timestamp', '')
            }

            # Dataset-specific fields
            if self._dataset_id == "sts":
                row.update({
                    "genre": r.get('genre', ''),
                    "filename": r.get('filename', ''),
                    "year": r.get('year', ''),
                    "human_score_0_to_5": r.get('human_score_0_to_5')
                })
            else:  # "para"
                row.update({"dataset": r.get('dataset', 'ParaNMT')})

            data.append(row)

        df = pd.DataFrame(data)

        # Use dataset-specific file names.
        results_path = output_path / f"{self._dataset_id}_results_{timestamp}.csv"
        df.to_csv(results_path, index=False)

        # Also save a summary metrics file.
        metrics = self.calculate_metrics(results)
        metrics_path = output_path / f"{self._dataset_id}_metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        self.log.info(f"Results saved to: {results_path}")
        self.log.info(f"Metrics saved to: {metrics_path}")

        return str(results_path)

    def print_summary(self, metrics: Dict[str, Any]):
        """Print a clean summary of results."""
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)

        if "error" in metrics:
            print(f"ERROR: {metrics['error']}")
            return

        print(f"Dataset: {self._dataset_id.upper()} ({self._score_scale} scale)")
        print(f"Success rate: {metrics['summary']['success_rate']:.1f}% ({metrics['summary']['successful_cases']}/{metrics['summary']['total_cases']})")
        print()

        print("CORRELATION (0-1 scale):")
        print(f"  Spearman: {metrics['correlation']['spearman']:.3f}")
        print(f"  Pearson:  {metrics['correlation']['pearson']:.3f}")
        print()

        print("ERROR METRICS (0-5 scale):")
        print(f"  Mean Abs Error:   {metrics['error_metrics']['mean_absolute_error']:.2f} stars")
        print(f"  Median Abs Error: {metrics['error_metrics']['median_absolute_error']:.2f} stars")
        print(f"  RMSE:             {metrics['error_metrics']['root_mean_squared_error']:.2f} stars")
        print(f"  Max Error:        {metrics['error_metrics']['max_absolute_error']:.2f} stars")
        print()

        print("ACCURACY (0-5 scale):")
        print(f"  Within ±0.5 stars: {metrics['accuracy_metrics']['within_0.5_stars_percentage']:.1f}%")
        print(f"  Within ±1.0 star:  {metrics['accuracy_metrics']['within_1_star_percentage']:.1f}%")
        print()

        print("SCORE DISTRIBUTION (0-1 scale):")
        print(f"  Benchmark  score mean:  {metrics['score_distribution']['human_mean']:.2f} ± {metrics['score_distribution']['human_std']:.2f}")
        print(f"  Calculated score mean:  {metrics['score_distribution']['rank_mean']:.2f} ± {metrics['score_distribution']['rank_std']:.2f}")
        print(f"  Benchmark  score range: [{metrics['score_distribution']['human_min']:.1f}, {metrics['score_distribution']['human_max']:.1f}]")
        print(f"  Calculated score range: [{metrics['score_distribution']['rank_min']:.1f}, {metrics['score_distribution']['rank_max']:.1f}]")

        # Performance assessment
        spearman = metrics['correlation']['spearman']
        print()
        print("PERFORMANCE ASSESSMENT:")
        if spearman > 0.85:
            print("  EXCELLENT (State-of-the-art performance)")
        elif spearman > 0.75:
            print("  GOOD (Solid performance)")
        elif spearman > 0.65:
            print("  FAIR (Room for improvement)")
        else:
            print("  POOR (Needs significant improvement)")

def setup_logging():
    """
    Setup logging to file and console.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    module_name = __name__
    log_file = os.path.join(LOG_DIR, f"{module_name}_{timestamp}.log")

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def generate_operations_mapping(operations: Tuple, start_weight: float = 0.6, step: float = 0.1):
    """
    Generate operation mappings by varying first operation's weight.
    Remaining weight is distributed among other operations with all possible splits.
    """
    operation_mappings = []
    num_ops = len(operations)

    if num_ops < 1:
        return []
    if num_ops == 1:
        return [((operations[0], 1.0),)]

    min_total_remaining = (num_ops - 1) * step

    if start_weight > 1.0 - min_total_remaining + 1e-10:
        raise ValueError(f"Cannot distribute weight: start_weight={start_weight}, "
                         f"need at least {min_total_remaining:.1f} for {num_ops - 1} other ops")

    cur_weight = start_weight

    while cur_weight <= 1.0 - min_total_remaining + 1e-10:
        remain_weight = 1.0 - cur_weight

        if num_ops == 2:
            operation_mappings.append((
                (operations[0], round(cur_weight, 10)),
                (operations[1], round(remain_weight, 10))
            ))
        else:
            other_ops = operations[1:]
            num_other = len(other_ops)

            def generate_splits(n: int, total: float, current: Tuple = ()):
                if n == 1:
                    if total >= step - 1e-10:
                        yield current + (round(total, 10),)
                else:
                    min_val = step
                    max_val = total - (n - 1) * min_val

                    if max_val >= min_val - 1e-10:
                        for first in np.arange(min_val, max_val + 1e-10, step):
                            first_rounded = round(first, 10)
                            for rest in generate_splits(n - 1, total - first_rounded, current + (first_rounded,)):
                                yield rest

            for split in generate_splits(num_other, remain_weight):
                mapping = [(operations[0], round(cur_weight, 10))]
                for i, weight in enumerate(split):
                    mapping.append((other_ops[i], weight))
                operation_mappings.append(tuple(mapping))

        cur_weight += step
        cur_weight = round(cur_weight, 10)

    return operation_mappings

# Main execution function
# Replace the main() function with:
async def main():
    logger = setup_logging()
    harness = BenchmarkHarness(DATASET_CHOICE, logger)

    best_spearman = -1
    best_operations = None
    best_metrics = None
    best_results = None

    try:
        # Load chosen dataset
        test_cases = harness.read_data()
        logger.info(f"Testing on {harness.dataset_id.upper()} dataset ({harness.score_scale} scale)")

        operation_sets = generate_operations_mapping(OPERATIONS, 0.6, 0.1)
        logger.info(f"Testing {len(operation_sets)} operation sets")

        for i, operations_case in enumerate(operation_sets):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Test {i + 1}/{len(operation_sets)}: {operations_case}")

            results = await harness.run_ranking_on_cases(test_cases,
                                                         lang_id="en",
                                                         operations=operations_case,
                                                         batch_size=100)

            metrics = harness.calculate_metrics(results)
            spearman = metrics['correlation']['spearman']

            logger.info(f"Spearman correlation: {spearman:.3f}")
            logger.info(f"Pearson correlation: {metrics['correlation']['pearson']:.3f}")

            if harness.dataset_id == "sts":
                logger.info(f"MAE: {metrics['error_metrics']['mean_absolute_error']:.3f} stars")
            else:
                # For 0-1 scale, convert MAE to 0-1
                mae_0_to_1 = metrics['error_metrics']['mean_absolute_error'] / 5.0
                logger.info(f"MAE: {mae_0_to_1:.3f} (0-1 scale)")

            if spearman > best_spearman:
                best_spearman = spearman
                best_operations = operations_case
                best_metrics = metrics
                best_results = results

        # Final results
        logger.info(f"\n{'=' * 60}")
        logger.info(f"FINAL BEST RESULTS for {harness.dataset_id.upper()} dataset")
        logger.info(f"{'=' * 60}")
        logger.info(f"Best operation set: {best_operations}")
        logger.info(f"Spearman correlation: {best_spearman:.3f}")

        if best_metrics:
            harness.print_summary(best_metrics)

        if best_results:
            path = harness.write_results(best_results)

    finally:
        rr_oper.EmbeddableRelevanceScoreOperation.flush_cache()


if __name__ == "__main__":
    # Run the test
    asyncio.run(main())
