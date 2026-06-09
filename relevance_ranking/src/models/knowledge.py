from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, Optional, Tuple

import utils as rr_utl


class KnowledgeInputType(Enum):
    """
    Type of operation input identifier, which is required
    to be passed for provided operation.
    """
    ANSWER = "answer"
    QUESTION = "question"


class SimilarityComparisonType(Enum):
    """
    Type of comparison identifier, which is requested
    to be performed for provided knowledge fragments.
    """
    DOCUMENT = "document"
    TOKEN = "token"


@dataclass(frozen=True)
class KnowledgeFragment:
    """
    Text representation (one or many sentences) with a lazy-loaded tokenization.

    Attributes:
        raw_fragment: Raw unformatted knowledge fragment.
        tokenizer: Utility, carrying specialized tokenization and pre-processing logic.

    Example of raw unformatted knowledge fragment: "knowledge_fragment": "<p> I lead the Enterprise AI team at TMNA.
        I am responsible for the overall architecture.</p>"
    """
    raw_fragment: str
    tokenizer: rr_utl.Tokenizer = field(repr=False, compare=False)

    # Internal cache representation.
    _fragment: Tuple[str, ...] = field(init=False, repr=False)
    _tokenized_fragment: Optional[Tuple[str, ...]] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """
        Validate and cache formatted fragment.
        """
        # Validate input type.
        if not isinstance(self.raw_fragment, str):
            raise TypeError(f"KnowledgeFragment: expected 'str', got '{type(self.raw_fragment).__name__}'")

        # Validate not empty.
        if not self.raw_fragment.strip():
            raise ValueError("KnowledgeFragment: input cannot be empty or whitespace only")

        # 3. Compute and cache formatted fragment.`
        formatted_value = self.raw_fragment.lower().strip()
        object.__setattr__(self, '_fragment', (formatted_value,))

    @property
    def fragment(self) -> Tuple[str, ...]:
        """
        Formatted and parsed (lowercase, stripped, etc.) string fragment.
        """
        return self._fragment

    @property
    def tokenized_fragment(self) -> Tuple[str, ...]:
        """
        Lazy-loads tokenized fragment using formatted fragment.
        """
        if self._tokenized_fragment is None:
            # Use cached fragment[0] directly
            tokenized_value = self.tokenizer.tokenize(self._fragment[0])
            object.__setattr__(self, '_tokenized_fragment', tokenized_value)

        return self._tokenized_fragment

    def attr_by_comparison_type(self, comparison_type: SimilarityComparisonType) -> Tuple[str]:
        """
        Depending on provided comparison type enumeration, return appropriate attribute
        (either fragment or tokenized fragment).

        Args:
            comparison_type (SimilarityComparisonType): Provided type enumeration.

        Returns:
            Tuple[str]: Appropriate attribute (either fragment or tokenized fragment).
        """
        target_attribute = ()
        if comparison_type == SimilarityComparisonType.DOCUMENT:
            target_attribute = self.fragment
        elif comparison_type == SimilarityComparisonType.TOKEN:
            target_attribute = self.tokenized_fragment

        return target_attribute


@dataclass(frozen=True)
class KnowledgeContainer:
    """
    A structured, immutable Q@A record, composed of knowledge fragments.

    Attributes:
        agent: The agent identifier that generated the container.
        answer: A KnowledgeFragment object representing the answer.
        question: A KnowledgeFragment object representing the question.
        metadata: An optional mapping for any other associated data components.

    Example of raw unformatted knowledge container:
    "knowledge_container": {
        "content": {
            "text": "Serge Feldman was asked: How do you contribute cross-functional collaborations in
            your work?\r\nSerge Feldman (TMNA) answered: Every department has a different role in our life cycle.
            Our organization invests alot into various tools."
        },
        "metadata": {
            "agent_id": "06745f234-efa23",
            "context_id": "34232fw-23w34"
        },
    }
    """
    agent: str
    answer: KnowledgeFragment
    question: KnowledgeFragment
    metadata: Dict[str, Any] = field(default_factory=dict)

    def attr_by_comparison_type(self, comparison_type: SimilarityComparisonType,
                                input_type: KnowledgeInputType = KnowledgeInputType.ANSWER) -> Tuple[str]:
        """
        Depending on provided comparison type enumeration, return appropriate attribute
        (either answer or question attribute).

        Args:
            comparison_type (SimilarityComparisonType): Provided comparison level type enumeration.
            input_type (KnowledgeInputType): Provided input type enumeration.

        Returns:
            Tuple[str]: Appropriate attribute (either answer or question attribute).
        """
        target_attribute = ()
        if comparison_type == SimilarityComparisonType.DOCUMENT:
            if input_type == KnowledgeInputType.ANSWER:
                target_attribute = self.answer.fragment
            elif input_type == KnowledgeInputType.QUESTION:
                target_attribute == self.question.fragment
        elif comparison_type == SimilarityComparisonType.TOKEN:
            if input_type == KnowledgeInputType.ANSWER:
                target_attribute = self.answer.tokenized_fragment
            elif input_type == KnowledgeInputType.QUESTION:
                target_attribute == self.question.tokenized_fragment

        return target_attribute


def create_knowledge_fragment(raw_string: str, tokenizer: rr_utl.Tokenizer) -> KnowledgeFragment:
    """
    Factory like function to parse a raw string and create a KnowledgeFragment object.

    Args:
        raw_string (str): Provided raw string containing knowledge fragment.
        tokenizer (rr_utl.Tokenizer): Provided Tokenizer utility.

    Returns:
        KnowledgeFragment: A fully constructed and validated KnowledgeFragment object.
    """
    # Validation.
    if not raw_string or not raw_string.strip():
        raise ValueError('Empty or malformed string input.')

    # Cleaning.
    clean_string = _strip_html(raw_string).strip()

    # Assembly.
    return KnowledgeFragment(clean_string, tokenizer)


def create_knowledge_container(raw_mapping: Dict, tokenizer: rr_utl.Tokenizer) -> KnowledgeContainer:
    """
    Factory like function to parse a raw mapping and create a KnowledgeContainer object.

    Args:
        raw_mapping (Dict): Provided raw input mapping containing knowledge context and metadata.
        tokenizer (rr_utl.Tokenizer): Provided Tokenizer utility.

    Returns:
        KnowledgeContainer: A fully constructed and validated KnowledgeContainer object.
    """
    # Validation.
    raw_content = raw_mapping.get('content', {})
    raw_text = raw_content.get('text')

    if not raw_text or not raw_text.strip():
        raise ValueError('Empty or malformed text input.')
    else:
        raw_text = raw_text.strip()

    metadata = raw_mapping.get('metadata', {})

    # Parsing.
    parts = re.split(r'\s+was asked:\s+', raw_text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) < 2:
        raise ValueError(f"Input raw string must contain both 'was asked:' and 'answered:'. ",
                         f"Found {len(parts)} occurrences.")
    raw_agent = parts[0]
    raw_qa = parts[1]

    answer_parts = re.split(r'\s+answered:\s+', raw_qa, maxsplit=1, flags=re.IGNORECASE)
    if len(answer_parts) < 2:
        raise ValueError(f"Could not find necessary answer components.")

    raw_question = answer_parts[0].splitlines()[0]
    raw_answer = answer_parts[1]

    # Cleaning.
    agent = _strip_html(raw_agent).strip()
    question = create_knowledge_fragment(raw_question, tokenizer)
    answer = create_knowledge_fragment(raw_answer, tokenizer)

    # Assembly.
    return KnowledgeContainer(agent, answer, question, metadata)


def _strip_html(input_string: str) -> str:
    """
    Removes HTML tags from provided string using regular expressions.

    Args:
        input_string (str): Provided string to remove HTML tags from.

    Returns:
        str: Generated string.
    """
    if not input_string or not input_string.strip():
        return ''

    html_pattern = re.compile(r'<.*?>')
    return re.sub(html_pattern, '', input_string)
