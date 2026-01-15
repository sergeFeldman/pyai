import pytest
from unittest.mock import Mock
import dataclasses

# Import from the models module.
from models import (
    KnowledgeFragment,
    KnowledgeContainer,
    create_knowledge_fragment,
    create_knowledge_unit
)


class TestKnowledgeFragment:
    """Test the KnowledgeFragment class"""

    def setup_method(self):
        # Create a mock tokenizer.
        self.mock_tokenizer = Mock()

    def test_knowledge_fragment_instantiation(self):
        """Test basic KnowledgeFragment instantiation"""
        fragment = KnowledgeFragment("Test fragment", self.mock_tokenizer)

        assert fragment.raw_fragment == "Test fragment"
        assert fragment.tokenizer == self.mock_tokenizer
        assert fragment._tokenized_fragment is None

    def test_knowledge_fragment_property(self):
        """Test the fragment property formatting"""
        # Test with whitespace.
        fragment = KnowledgeFragment("  TEST Fragment  ", self.mock_tokenizer)
        assert fragment.fragment == ("test fragment",)

        # Test with uppercase.
        fragment = KnowledgeFragment("UPPERCASE TEXT", self.mock_tokenizer)
        assert fragment.fragment == ("uppercase text",)

    def test_knowledge_fragment_tokenized(self):
        """Test lazy loading of tokenized fragment"""
        fragment = KnowledgeFragment("Test fragment", self.mock_tokenizer)

        # Cache should be empty initially.
        assert fragment._tokenized_fragment is None
        assert self.mock_tokenizer.tokenize.call_count == 0

        # First access should trigger tokenization.
        self.mock_tokenizer.tokenize.return_value = ("token1", "token2", "token3")
        result = fragment.tokenized_fragment
        assert self.mock_tokenizer.tokenize.call_count == 1
        assert result == ("token1", "token2", "token3")
        assert fragment._tokenized_fragment == result

        # Second access should use cache.
        result2 = fragment.tokenized_fragment
        assert self.mock_tokenizer.tokenize.call_count == 1
        assert result2 == result

        # Verify tokenizer was called with the fragment text.
        self.mock_tokenizer.tokenize.assert_called_with("test fragment")

    def test_knowledge_fragment_erroneous_validation(self):
        """Test that empty fragments raise ValueError"""
        mock_tokenizer = Mock()

        # Incorrect type should raise.
        with pytest.raises(TypeError, match="KnowledgeFragment: expected 'str', got 'int'"):
            KnowledgeFragment(123, mock_tokenizer)

        # Empty string should raise.
        with pytest.raises(ValueError, match="cannot be empty"):
            KnowledgeFragment("", mock_tokenizer)

        # Whitespace only should raise.
        with pytest.raises(ValueError, match="cannot be empty"):
            KnowledgeFragment("   ", mock_tokenizer)

        # Tabs/newlines only should raise.
        with pytest.raises(ValueError, match="cannot be empty"):
            KnowledgeFragment("\t\n\r", mock_tokenizer)

        # Valid fragment with content should work.
        fragment = KnowledgeFragment("Valid text", mock_tokenizer)
        assert fragment.fragment == ("valid text",)

    def test_knowledge_fragment_equality(self):
        """Test equality comparison of KnowledgeFragment instances"""
        tokenizer1 = Mock()
        tokenizer2 = Mock()

        fragment1 = KnowledgeFragment("Same text", tokenizer1)
        fragment2 = KnowledgeFragment("Same text", tokenizer2)
        fragment3 = KnowledgeFragment("Different text", tokenizer1)

        # Should be equal based on raw_fragment only.
        assert fragment1 == fragment2
        assert fragment1 != fragment3


class TestCreateKnowledgeFragment:
    """Test create_knowledge_fragment() factory function"""

    def setup_method(self):
        # Create a mock tokenizer.
        self.mock_tokenizer = Mock()

    def test_create_knowledge_fragment_valid(self):
        """Test creating a KnowledgeFragment with valid input"""

        # Create fragment using factory
        fragment = create_knowledge_fragment("  <p>Test fragment</p>  ", self.mock_tokenizer)

        # Verify fragment properties
        assert fragment.raw_fragment == "Test fragment"  # HTML stripped and trimmed
        assert fragment.fragment == ("test fragment",)
        assert fragment.tokenizer == self.mock_tokenizer

    def test_create_knowledge_fragment_erroneous_validation(self):
        """Test creating KnowledgeFragment with empty string raises error"""

        with pytest.raises(ValueError, match="Empty or malformed string input"):
            create_knowledge_fragment("", self.mock_tokenizer)

        with pytest.raises(ValueError, match="Empty or malformed string input"):
            create_knowledge_fragment("   ", self.mock_tokenizer)

        with pytest.raises(ValueError, match="KnowledgeFragment: input cannot be empty or whitespace only"):
            create_knowledge_fragment("<p></p>", self.mock_tokenizer)


class TestKnowledgeUnit:
    """Test the KnowledgeContainer class"""

    def setup_method(self):
        # Create mock tokenizer and fragments.
        self.mock_tokenizer = Mock()
        self.question = KnowledgeFragment("What is AI?", self.mock_tokenizer)
        self.answer = KnowledgeFragment("AI is Artificial Intelligence", self.mock_tokenizer)
        self.metadata = {"source": "test", "id": 123}

    def test_knowledge_unit_instantiation(self):
        """Test basic KnowledgeContainer instantiation"""
        test_unit = KnowledgeContainer(
            agent="TestAgent", metadata=self.metadata,
            question=self.question, answer=self.answer)

        assert test_unit.agent == "TestAgent"
        assert test_unit.question == self.question
        assert test_unit.answer == self.answer
        assert test_unit.metadata == self.metadata

        test_unit = KnowledgeContainer(
            agent="TestAgent",
            question=self.question, answer=self.answer )

        assert test_unit.metadata == {}

    def test_knowledge_unit_immutability(self):
        """Test that KnowledgeContainer is immutable"""
        test_unit = KnowledgeContainer("Agent", self.answer, self.question, self.metadata)

        with pytest.raises(dataclasses.FrozenInstanceError):
            test_unit.agent = "NewAgent"

        with pytest.raises(dataclasses.FrozenInstanceError):
            test_unit.metadata = {"source": "test", "id": 123}


class TestCreateKnowledgeUnit:
    """Test create_knowledge_unit() factory function"""

    def setup_method(self):
        # Create mock tokenizer and fragments
        self.mock_tokenizer = Mock()
        self.raw_mapping = {
            'content': {
                'text': "Serge Feldman was asked: How do you contribute cross-functional collaborations?\r\nSerge Feldman (TMNA) answered: Every department has a different role."
            },
            'metadata': {
                'agent_id': '06745f234-efa23',
                'context_id': '34232fw-23w34'
            }
        }

    def test_create_knowledge_unit_valid(self):
        """Test creating KnowledgeContainer with valid input"""

        test_unit = create_knowledge_unit(self.raw_mapping, self.mock_tokenizer)

        # Verify unit properties.
        assert test_unit.agent == "Serge Feldman"
        assert isinstance(test_unit.question, KnowledgeFragment)
        assert test_unit.question.raw_fragment == "How do you contribute cross-functional collaborations?"
        assert isinstance(test_unit.answer, KnowledgeFragment)
        assert test_unit.answer.raw_fragment == "Every department has a different role."
        assert test_unit.metadata['agent_id'] == '06745f234-efa23'

    def test_create_knowledge_unit_erroneous_validation(self):
        """Test creating KnowledgeContainer with missing content"""
        raw_mapping_erroneous_context_empty = {'content': {'text': '    '}}
        raw_mapping_erroneous_was_asked = {'content': {'text': "Just some text without expected format"}}
        raw_mapping_erroneous_answer = {'content': {'text': "Serge Feldman was asked: Just a question without answer"}}

        with pytest.raises(ValueError, match="Empty or malformed text input"):
            create_knowledge_unit(raw_mapping_erroneous_context_empty, self.mock_tokenizer)

        with pytest.raises(ValueError) as exc_info:
            create_knowledge_unit(raw_mapping_erroneous_was_asked, self.mock_tokenizer)

        assert "must contain both 'was asked:' and 'answered:'" in str(exc_info.value)
        assert "Found 1 occurrences" in str(exc_info.value)

        with pytest.raises(ValueError, match="Could not find necessary answer components"):
            create_knowledge_unit(raw_mapping_erroneous_answer, self.mock_tokenizer)

    def test_create_knowledge_unit_with_html(self):
        """Test creating KnowledgeContainer with HTML in text"""
        raw_mapping = {
            'content': {
                'text': "<b>Serge Feldman</b> was asked: <p>How do you contribute?</p>\r\nSerge Feldman (TMNA) answered: <div>Every department has a role.</div>"
            },
        }

        test_unit = create_knowledge_unit(raw_mapping, self.mock_tokenizer)

        # Agent should have HTML stripped
        assert test_unit.agent == "Serge Feldman"
        assert isinstance(test_unit.question, KnowledgeFragment)
        assert test_unit.question.raw_fragment == "How do you contribute?"
        assert isinstance(test_unit.question, KnowledgeFragment)
        assert test_unit.answer.raw_fragment == "Every department has a role."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])