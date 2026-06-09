from logging import Logger
import asyncio
import math
from typing import Dict, Tuple, Union

import core as rr_core
import models as rr_models
import operations as rr_oper
import utils as rr_utl

PRIMARY_OPERATIONS_COUNT = 1   # Define operation tiers.
PRIMARY_SCORE_THRESHOLD = 0.20 # 20% threshold for primary operations scoring.
TOKEN_SCORE_THRESHOLD = 0.20   # 20% threshold for TOKEN-level attributes scoring.

# Default mapping of ranking operations, which can be engaged
# via orchestration workflow.
# Note: The requirement is to place the primary operations at the beginning.
DEFAULT_OPERATIONS_MAPPING: Tuple = (
    (rr_oper.SemanticSimilarityOperation, 0.6),
    (rr_oper.FuzzySimilarityOperation, 0.4)
)


async def run_ranking_async(log: Logger,
                            raw_data_obj: Union[Dict, str],
                            raw_data_ref_obj: str,
                            lang_id: str = "en",
                            operations: Tuple = DEFAULT_OPERATIONS_MAPPING) -> Dict:
    """
    Main entry point orchestration for ranking score calculation operations,
    implemented as direct async function with concurrent processing.

    Run the calculation of DOCUMENT-level attributes first. If the overall score
    is above provided threshold, continue with the TOKEN-level attribution execution.
    Otherwise, stop and skip all the consecutive operations, thus saving time and
    performance execution.
    Return generated relevance scores matrix.

    Args:
        log (Logger): Provided logger.
        raw_data_obj (Dict, or str): Provided raw variable input structure.
        raw_data_ref_obj (str): Provided raw compare-to baseline input structure.
        lang_id (str, default="en"): Language identifier.
        operations (Tuple, default=DEFAULT_OPERATIONS_MAPPING): Provided mapping
            of operations, which the orchestrator will iterate over.
            Once again, primary operations are placed at the beginning of the mapping.

    Returns:
        Dict: Generated relevance scores matrix.
    """
    rr_core.BaseLoggable.set_logger(log)

    try:

        # Phase 0: Validation of provided configuration.
        _validate_config(operations, PRIMARY_OPERATIONS_COUNT)

        # Phase 1: Pre-processing / normalization / formatting.

        # Tokenizer instantiation.
        tokenizer = rr_utl.Tokenizer(lang_id)

        if isinstance(raw_data_obj, str):
            knowledge_obj = rr_models.create_knowledge_fragment(raw_data_obj, tokenizer)
        else:
            knowledge_obj = rr_models.create_knowledge_container(raw_data_obj, tokenizer)

        knowledge_ref_obj = rr_models.create_knowledge_fragment(raw_data_ref_obj, tokenizer)

        # Phase 2: Document-level processing.
        doc_scores = await _calculate_ranking_async(log, rr_models.SimilarityComparisonType.DOCUMENT,
                                                    knowledge_obj, knowledge_ref_obj,
                                                    operations, PRIMARY_OPERATIONS_COUNT,
                                                    PRIMARY_SCORE_THRESHOLD)

        # Phase 3: Token-level processing (conditionally).
        tkn_scores = {}
        # Check if document-level primary score meets threshold for token-level processing.
        # Note: For doc-level, we assume single fragment comparison, so we check score at index "0":"0"
        # This works because doc_scores structure is always {"0": {"0": score}} for document-level
        primary_doc_weighted_score = doc_scores.get("0", {}).get("0", 0.0)
        primary_doc_score = primary_doc_weighted_score / operations[0][1] if operations[0][1] > 0 else 0.0

        if primary_doc_score >= PRIMARY_SCORE_THRESHOLD:
            # Check if fragments have single tokens/sentences.
            if (len(knowledge_obj.attr_by_comparison_type(rr_models.SimilarityComparisonType.TOKEN)) == 1 and
                len(knowledge_ref_obj.attr_by_comparison_type(rr_models.SimilarityComparisonType.TOKEN)) == 1):
                # Single tokens: token-level = document-level.
                tkn_scores = doc_scores
                log.debug("Token-level scores copied from document-level (single tokens)")
            else:
                tkn_scores = await _calculate_ranking_async(log, rr_models.SimilarityComparisonType.TOKEN,
                                                            knowledge_obj, knowledge_ref_obj,
                                                            operations, PRIMARY_OPERATIONS_COUNT,
                                                            TOKEN_SCORE_THRESHOLD)

        return {
            "rank_score": {
                "doc_level": doc_scores,
                "tkn_level": tkn_scores
            }
        }
    except (ValueError, TypeError) as e:
        log.error(f"Input validation error in knowledge processing: {str(e)}")
        raise
    except Exception as e:
        log.error(f"Unexpected error in ranking orchestration: {str(e)}")
        raise


def run_ranking_sync(log: Logger,
                     raw_data_obj: Union[Dict, str],
                     raw_data_ref_obj: str,
                     lang_id: str = "en",
                     operations: Tuple = DEFAULT_OPERATIONS_MAPPING) -> Dict:
    """
    Synchronous wrapper for run_ranking_async() function for enhanced processing
    using async orchestration.

    Args:
        log (Logger): Provided logger.
        raw_data_obj (Dict, or str): Provided raw variable input structure.
        raw_data_ref_obj (str): Provided raw compare-to baseline input structure.
        lang_id (str, default="en"): Language identifier.
        operations (Tuple, default=DEFAULT_OPERATIONS_MAPPING): Provided mapping
            of operations, which the orchestrator will iterate over.

    Returns:
        Dict: Generated relevance scores matrix.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        return loop.run_until_complete(
            run_ranking_async(log, raw_data_obj, raw_data_ref_obj, lang_id, operations)
        )
    finally:
        loop.close()


async def _calculate_ranking_async(log: Logger,
                                   comparison_type: rr_models.SimilarityComparisonType,
                                   knowledge_obj: Union[rr_models.KnowledgeContainer, rr_models.KnowledgeFragment],
                                   knowledge_ref_obj: rr_models.KnowledgeFragment,
                                   operations: Tuple,
                                   primary_operations_count: int,
                                   score_threshold: float) -> Dict:
    """
    Wrapper for ranking score calculation, given provided knowledge unit and
    compare-to fragment objects.

    The logic is improved to take primary/main operation threshold into account.
    Run the calculation of primary operation(s) first. If the overall primary score
    is above provided threshold, continue with the rest of execution.
    Otherwise, skip all the consecutive operations, thus eliminating unnecessary
    process execution.

    Args:
        log (Logger): Provided logger.
        comparison_type (rr_models.SimilarityComparisonType): Provided comparison type
            (doc-level and token-level).
        knowledge_obj (rr_models.KnowledgeContainer or rr_models.KnowledgeFragment): Provided knowledge query object.
        knowledge_ref_obj (rr_models.KnowledgeFragment): Provided knowledge compare-to baseline fragment object.
        operations (Tuple): Provided mapping of operations.
        primary_operations_count (int): Number of primary operations, contained in the mapping.
        score_threshold (float): Threshold for combined score of primary operation(s) calculation.

    Returns:
        Dict: Nested dictionary of weighted scores {idx: {ref_idx: weighted_score}}
    """
    # Initialize target scores.
    target_scores_mapping = {}

    # Extract text fragments from knowledge object parameters.
    text_fragments = knowledge_obj.attr_by_comparison_type(comparison_type)
    text_ref_fragments = knowledge_ref_obj.attr_by_comparison_type(comparison_type)

    for idx, text_fragment in enumerate(text_fragments):
        target_scores_mapping[str(idx)] = {}

        for ref_idx, text_ref_fragment in enumerate(text_ref_fragments):
            # Safety check for empty fragments.
            if text_fragment and text_ref_fragment:
                primary_score, primary_weighted_score = \
                    await _calculate_operation_ranking_async(log,
                                                             operations[:primary_operations_count],
                                                             text_fragment,
                                                             text_ref_fragment)

                # Initialize secondary scores.
                secondary_score, secondary_weighted_score = 0.0, 0.0
                # Examine the sum of calculated primary scores against provided threshold
                # (make sure to use non-weighted score).
                if primary_score >= score_threshold:

                    secondary_score, secondary_weighted_score = \
                        await _calculate_operation_ranking_async(log,
                                                                 operations[primary_operations_count:],
                                                                 text_fragment,
                                                                 text_ref_fragment)

                # Accumulate scores from all operations for given comparison pair.
                target_score = primary_score + secondary_score
                target_weighted_score = primary_weighted_score + secondary_weighted_score
                target_scores_mapping[str(idx)][str(ref_idx)] = target_weighted_score

                log.debug("async calculate ranking completed",
                          extra={"text_fragment": text_fragment,
                                 "text_ref_fragment": text_ref_fragment,
                                 "score": f"{primary_score} + {secondary_score}",
                                 "weighted score": f"{primary_weighted_score} + {secondary_weighted_score}"})
            else:
                log.warning("Empty text fragments to compare",
                            extra={"text_fragment_len": len(text_fragment),
                                   "text_ref_fragment_len": len(text_ref_fragment)})
                target_scores_mapping[str(idx)][str(ref_idx)] = 0.0
    return target_scores_mapping


async def _calculate_operation_ranking_async(log: Logger,
                                             operations: Tuple,
                                             text_fragment: str,
                                             text_ref_fragment: str) -> Tuple[float, float]:
    """
    Process the calculation of ranking scores for multiple provided
    operations and combine their scores into a unified score.

    Args:
        log (Logger): Provided logger.
        operations (Tuple): Provided mapping of operations.
        text_fragment (str): Provided text query fragment.
        text_ref_fragment (str): Provided text compare-to baseline fragment.

    Returns:
        Tuple: Collection of calculated scores and weighted scores.
    """
    # Run all operations concurrently.
    loop = asyncio.get_event_loop()
    tasks = []

    for operation_context in operations:
        operation_type = operation_context[0]
        operation_weight = operation_context[1]
        task = loop.run_in_executor(None, operation_type.calculate_weighted,
                                    text_fragment,
                                    text_ref_fragment,
                                    operation_weight)
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    scores = []
    weighted_scores = []

    for i, operation_context in enumerate(operations):
        operation_type = operation_context[0]
        operation_weight = operation_context[1]

        try:
            if isinstance(results[i], Exception):
                log.error(f"Operation {operation_type.__name__} failed: {results[i]}")
                scores.append(0.0)
                weighted_scores.append(0.0)
            else:
                weighted_score = results[i]
                score = weighted_score / operation_weight if operation_weight > 0 else 0.0
                scores.append(score)
                weighted_scores.append(weighted_score)

        except Exception as e:
            log.error(f"Error processing operation {operation_type.__name__} result: {str(e)}")
            scores.append(0.0)
            weighted_scores.append(0.0)

    log.debug("async operation ranking calculation completed",
              extra={"operations": [op[0].__name__ for op in operations],
                     "text_fragment": text_fragment,
                     "text_ref_fragment": text_ref_fragment,
                     "scores": scores,
                     "weighted_scores": weighted_scores})

    return sum(scores), sum(weighted_scores)


def _validate_config(operations: Tuple, primary_operations_count: int) -> None:
    """
    Validate ranking configuration parameters.

    Args:
        operations: Tuple of (operation_class, weight) pairs
        primary_operations_count: Number of primary operations

    Raises:
        ValueError: If any validation check fails with descriptive error message.
    """
    # 1. Basic structure checks.
    if not operations:
        raise ValueError("At least one operation is required")

    if not isinstance(operations, tuple):
        raise ValueError(f"Operations must be a tuple, got {type(operations).__name__}")

    if primary_operations_count < 1:
        raise ValueError(f"primary_operations_count must be at least 1, got {primary_operations_count}")

    # 2. Validate each operation
    weights = []
    for i, op in enumerate(operations):
        # Check tuple structure
        if not isinstance(op, tuple) or len(op) != 2:
            raise ValueError(
                f"Operation at index {i} must be a (operation_class, weight) tuple, "
                f"got {type(op).__name__} with length {len(op) if isinstance(op, tuple) else 'N/A'}")

        op_class, weight = op

        # Check weight is numeric.
        if not isinstance(weight, (int, float)):
            raise ValueError(
                f"Operation at index {i}: weight must be numeric, "
                f"got {type(weight).__name__} ({weight})")

        # Check weight range.
        if weight < 0:
            raise ValueError(f"Operation at index {i}: weight must be non-negative, got {weight}")

        weights.append(float(weight))

    # 3. Validate weights sum.
    total_weight = sum(weights)

    # Some float values don't sum correctly due to binary representation limits.
    # Because of that, math.isclose() is used.
    if not math.isclose(total_weight, 1.0, rel_tol=1e-9):
        raise ValueError(f"Operation weights must sum to 1.0, got {total_weight:.2f}. "
                         f"Weights: {[f'{w:.2f}' for w in weights]}")

    # 4. Validate primary operations count.
    if primary_operations_count > len(operations):
        raise ValueError(f"primary_ops_count ({primary_operations_count}) cannot exceed "
                         f"total operations count ({len(operations)})")

    # 5. Validate primary operations have positive total weight.
    primary_weight_sum = sum(weights[:primary_operations_count])
    if primary_weight_sum <= 0:
        raise ValueError(f"Primary operations must have positive total weight, got {primary_weight_sum}")
