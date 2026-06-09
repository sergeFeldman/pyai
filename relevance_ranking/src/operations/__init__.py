"""
The operation package provides concrete relevance score calculation operations.
"""
from operations.base import EmbeddableRelevanceScoreOperation

from operations.similarity import (
    RelevanceScoreOperation,
    EntityOverlapSimilarityOperation,
    FuzzySimilarityOperation,
    SemanticSimilarityOperation
)

__all__ = [
    'EmbeddableRelevanceScoreOperation',
    'EntityOverlapSimilarityOperation',
    'RelevanceScoreOperation',
    'FuzzySimilarityOperation',
    'SemanticSimilarityOperation'
]
