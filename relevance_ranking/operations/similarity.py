import math

from markdown_it.common.entities import entities
from rapidfuzz import fuzz
from typing import List

import utils as rr_utl

from .base import (RelevanceScoreOperation,
                   RelevanceScoreOperationWithBedrock)


class SemanticSimilarityOperation(RelevanceScoreOperationWithBedrock):
    """
    Concrete static class, which implements
    "Semantic similarity" with specific embedding model logic.
    Note: Calculation with completely unrelated strings will
    produce a small positive score value and that is normal and expected.
    This reflects the mathematical reality that in NLP texts share common
    linguistic elements even when they are semantically unrelated.
    """

    @staticmethod
    def calculate(text_fragment1: str, text_fragment2: str) -> float:
        """
        Calculates the semantic similarity between two provided strings.

        Args:
            text_fragment1 (str): Provided text fragment, used in calculation.
            text_fragment2 (str): Provided text fragment to compare to, used in calculation.

        Returns:
            float: Calculated value.
        """
        try:

            text_embedding1 = SemanticSimilarityOperation.embed(text_fragment1)
            text_embedding2 = SemanticSimilarityOperation.embed(text_fragment2)

            if not text_embedding1 or not text_embedding2:
                return 0.0

            similarity = SemanticSimilarityOperation._cosine_similarity(text_embedding1, text_embedding2)
            return min(1.0, max(0.0, similarity))
        except Exception as e:
            SemanticSimilarityOperation.logger().error(f"Semantic similarity error: {str(e)}")
            return 0.0

    @staticmethod
    def _cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculates the cosine similarity between two provided vectors.

        Args:
            embedding1 (List[float]): First vector, used in calculation.
            embedding2 (List[float]): Second vector, used in calculation.

        Returns:
            float: Calculated value.
        """
        try:
            dot_product = sum(elem1 * elem2 for elem1, elem2 in zip(embedding1, embedding2))
            norm1 = math.sqrt(sum(elem1 * elem1 for elem1 in embedding1))
            norm2 = math.sqrt(sum(elem2 * elem2 for elem2 in embedding2))

            return dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0.0
        except Exception as e:
            SemanticSimilarityOperation.logger().error(f"Cosine similarity calculation error: {str(e)}")
            return 0.0


class FuzzySimilarityOperation(RelevanceScoreOperation):
    """
    Concrete static class, which implements
    "Fuzzy similarity matching" logic, using rapidFuzz library.
    """

    @staticmethod
    def calculate(text_fragment1: str, text_fragment2: str) -> float:
        """
        Calculates the raw score between two provided strings,
        using rapidFuzz library.

        Args:
            text_fragment1 (str): First text fragment, used in calculation.
            text_fragment2 (str): Text fragment to compare to, used in calculation.

        Returns:
            float: Calculated value.
        """
        try:
            return fuzz.ratio(text_fragment1, text_fragment2) / 100.0
        except Exception as e:
            FuzzySimilarityOperation.logger().error(f"Fuzzy operation error: {str(e)}")
            return 0.0


class EntityOverlapSimilarityOperation(RelevanceScoreOperation):
    """
    Concrete static class, which implements spacy logic.
    """

    @staticmethod
    def calculate(text_fragment1: str, text_fragment2: str) -> float:
        """
        Calculates the raw score between two provided strings, using spacy logic,

        Args:
            text_fragment1 (str): First text fragment, used in calculation.
            text_fragment2 (str): Text fragment to compare to, used in calculation.

        Returns:
            float: Calculated value.
        """
        try:
            tokenizer = rr_utl.Tokenizer('en')
            entities1 = EntityOverlapSimilarityOperation._extract_entities(text_fragment1, tokenizer)
            entities2 = EntityOverlapSimilarityOperation._extract_entities(text_fragment2, tokenizer)

            if not entities1 or not entities2:
                return 0.0

            intersection = len(entities1 & entities2)
            union = len(entities1 | entities2)

            overlap_ratio = intersection / union if union > 0.0 else 0.0
            return min(1.0, max(0.0, overlap_ratio))
        except Exception as e:
            EntityOverlapSimilarityOperation.logger().error(f"Entity overlap calculation error: {str(e)}")
            return 0.0

    @staticmethod
    def _extract_entities(text_fragment: str, tokenizer: rr_utl.Tokenizer) -> set:
        """
        Extracts entities, using spacy.

        Args:
            text_fragment (str): Text fragment, used in calculation.
            tokenizer (Tokenizer): Spacy Tokenizer.

        Returns:
            set: Generated collection of strings
        """
        if tokenizer._nlp:
            doc = tokenizer._nlp(text_fragment[:5000])
            return set([entry.text.lower() for entry in doc.ents])
        else:
            return set()
