import spacy
from typing import Dict, Optional
import core as rr_core

class SpaCyManager(rr_core.BaseLoggable):
    """
    A dedicated manager for spaCy NLP models.
    Provides a utility interface for loading and caching models for different languages.
    Loaded spaCy models are cached at the class level to prevent redundant loading.
    """
    # Static constant mapping of language IDs to supported spaCy model names.
    _DEFAULT_MODEL_MAPPING: Dict[str, str] = {
        'en': 'en_core_web_sm',
        'jp': 'ja_core_news_sm'
    }

    # Cache of loaded spaCy models, indexed by language ID
    # (to ensure each model is loaded only once).
    _nlp_mapping: Dict[str, Optional[spacy.Language]] = {}

    @classmethod
    def get_model(cls, lang_id: str) -> Optional[spacy.Language]:
        """
        Retrieve or load a spaCy language model for the given language ID.
        Implements a lazy loading with caching. If the model for requested language
        hasn't been loaded yet, it will attempt to load it and cache the result.

        Args:
            lang_id (str): The language identifier. Must be a key in _DEFAULT_MODEL_MAPPING.

        Returns:
            spacy.Language object, or None: The loaded (or cached) spaCy model if successful,
                                            None, if loading fails, or language unsupported.

        Raises:
            ValueError: If provided lang_id is not supported (not in _DEFAULT_MODEL_MAPPING).
        """
        if lang_id not in cls._DEFAULT_MODEL_MAPPING:
            raise ValueError(
                f"Unsupported language ID '{lang_id}'. "
                f"Supported languages: {list(cls._DEFAULT_MODEL_MAPPING.keys())}")

        if lang_id not in cls._nlp_mapping:
            cls._load_nlp_model(lang_id)

        return cls._nlp_mapping[lang_id]

    @classmethod
    def _load_nlp_model(cls, lang_id) -> None:
        """
        Loads and caches the spaCy language model for the instance's language ID.

        Args:
            lang_id (str): The language identifier. Must be a key in _DEFAULT_MODEL_MAPPING.

        Returns:
            None.
        """
        try:
            model_name = cls._DEFAULT_MODEL_MAPPING[lang_id]
            # Use the inherited logger from BaseLoggable.
            cls.logger().info(f"Loading spaCy model '{model_name}'")
            cls._nlp_mapping[lang_id] = spacy.load(model_name)
        except (OSError, ValueError) as e:
            cls.logger().error(f"Failed to load spaCy model '{lang_id}'. Error: {str(e)}")
            cls._nlp_mapping[lang_id] = None
