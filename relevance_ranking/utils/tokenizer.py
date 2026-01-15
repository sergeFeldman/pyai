import re
from typing import Tuple

import core as rr_core
from . import SpaCyManager


class Tokenizer(rr_core.BaseLoggable):
    """
    A dedicated class for text tokenization that splits text into tokens (sentences).
    Furthermore, this utility can also preprocess provided text.

    It primarily uses spaCy for robust linguistic tokenization and preprocessing and
    falls back to a simpler regex-based method if a spaCy model is unavailable or fails.
    """
    def __init__(self, lang_id: str):
        """
        Initialize a Tokenizer instance for a specific language.

        Args:
            lang_id (str): The language identifier.

        Raises:
            ValueError: If provided lang_id is not supported.
        """
        self._lang_id = lang_id
        self._nlp = SpaCyManager.get_model(lang_id)

    @property
    def lang_id(self) -> str:
        """
        Accessor for the language ID attribute.
        """
        return self._lang_id

    def preprocess(self, text: str,
                   lowercase: bool = True,
                   remove_stopwords: bool = True,
                   remove_punctuation: bool = True,
                   lemmatize: bool = True,
                   remove_nums: bool = False) -> str:
        """
        Pre-processes an input string into a cleaned/normalized string.

        Args:
            text (str): Input string.
            lowercase: Convert to lowercase
            remove_stopwords: Remove stop words (the, a, an, etc.)
            remove_punctuation: Remove punctuation marks
            lemmatize: Convert words to base form (running → run)
            remove_nums: Remove numeric characters

        Returns:
            str: Cleaned/normalized string.
        """
        # Prefer spaCy for preprocessing if the model was loaded successfully.
        if self._nlp:
            try:
                doc = self._nlp(text)
                # Filter / preprocess.
                tokens = [
                    self._preprocess_token(token, lowercase, lemmatize)
                    for token in doc if not self._should_skip_token(token, remove_stopwords,
                                                                    remove_punctuation,remove_nums)
                ]
                return " ".join(tokens).strip()
            except Exception as e:
                self.logger().error(f"spaCy preprocessing error: {str(e)}")

        # Fallback to regex if spaCy is not available or fails.
        return self._preprocess_regex(text, lowercase, remove_punctuation, remove_nums)

    def tokenize(self, text: str) -> Tuple[str, ...]:
        """
        Tokenizes an input string into a collections of string tokens (sentences).

        Args:
            text (str): Input string to be tokenized.

        Returns:
            A tuple of tokens (sentences).
        """
        # Prefer spaCy for tokenization if the model was loaded successfully.
        if self._nlp:
            try:
                doc = self._nlp(text)
                tokens = [token.text.strip() for token in doc.sents if token.text.strip()]
                # if tokenization results in an empty list,
                # return the original text as a single-element tuple.
                return tuple(tokens) if tokens else (text,)
            except Exception as e:
                self.logger().error(f"spaCy processing error, falling back to regex tokenizer: {e}")

        # Fallback to regex if spaCy is not available or fails.
        return self._tokenize_regex(text)

    def tokenize_and_preprocess(self, text: str,
                                lowercase: bool = True,
                                remove_stopwords: bool = True,
                                remove_punctuation: bool = True,
                                lemmatize: bool = True,
                                remove_nums: bool = False) -> Tuple[str, ...]:
        """
        Tokenizes an input string into a collections of string tokens (sentences) and
        then normalizes each token.

        Args:
            text (str): Input string.
            lowercase: Convert to lowercase
            remove_stopwords: Remove stop words
            remove_punctuation: Remove punctuation marks
            lemmatize: Convert words to base form
            remove_nums: Remove numeric characters

        Returns:
            A tuple of normalized preprocessed tokens (sentences).
        """
        return tuple(
            preprocessed
            for token in self.tokenize(text)
            if (preprocessed := self.preprocess(
                token,
                lowercase=lowercase,
                remove_stopwords=remove_stopwords,
                remove_punctuation=remove_punctuation,
                lemmatize=lemmatize,
                remove_nums=remove_nums)
            )
        )

    def _should_skip_token(self, token,
                           remove_stopwords: bool,
                           remove_punctuation: bool,
                           remove_nums: bool) -> bool:
        """
        Check if token should be skipped during preprocessing.

        Args:
            token: spaCy token object.
            remove_stopwords: Remove stop words (the, a, an, etc.)
            remove_punctuation: Remove punctuation marks
            remove_nums: Remove numeric characters

        Returns:
            bool: Result of validation.
        """
        return (
            (remove_stopwords and token.is_stop) or
            (remove_punctuation and token.is_punct) or
            (remove_nums and token.like_num)
        )

    def _preprocess_token(self, token, lowercase: bool, lemmatize: bool) -> str:
        # Get base form
        token_text = token.lemma_ if lemmatize and token.lemma_ else token.text

        # Apply lowercase
        if lowercase:
            token_text = token_text.lower()

        return token_text

    def _preprocess_regex(self, text: str,
                          lowercase: bool,
                          remove_punctuation: bool,
                          remove_nums: bool) -> str:
        """Simple regex cleaning."""
        if lowercase and self.lang_id == 'en':
            text = text.lower()  # Only English gets lowercased

        if remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)  # Removes ALL punctuation

        if remove_nums:
            text = re.sub(r'[\d０-９]+', ' ', text)  # Both standard & Japanese numbers

        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _tokenize_regex(self, text: str) -> Tuple[str, ...]:
        """
        A fallback tokenization method using regular expressions.

        Args:
            text (str): Input string to be tokenized.

        Returns:
            A tuple of tokens (sentences).
        """
        if self.lang_id == 'jp':
            # Japanese punctuation
            lang_pattern = r'[。！？]+'
        else:
            lang_pattern = r'[.!?]+(?:\s+|$)'

        parts = re.split(lang_pattern, text)
        return tuple(part.strip() for part in parts if part and part.strip())
