import pytest
from unittest.mock import Mock, patch
import utils as rr_utl


class TestTokenizer:
    """Test the Tokenizer class"""

    def setup_method(self):
        """Clear spaCy cache before each test"""
        rr_utl.SpaCyManager._nlp_mapping.clear()

    @patch('spacy.load')
    def test_tokenizer_initialization_valid(self, mock_spacy_load):
        """Test Tokenizer initialization with supported languages"""
        mock_nlp = Mock()
        mock_spacy_load.return_value = mock_nlp

        # English
        tokenizer_en = rr_utl.Tokenizer("en")
        assert tokenizer_en.lang_id == "en"
        mock_spacy_load.assert_called_once_with("en_core_web_sm")

        # Japanese
        mock_spacy_load.reset_mock()
        tokenizer_jp = rr_utl.Tokenizer("jp")
        assert tokenizer_jp.lang_id == "jp"
        mock_spacy_load.assert_called_once_with("ja_core_news_sm")

    def test_tokenizer_initialization_invalid(self):
        """Test Tokenizer with unsupported language raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            rr_utl.Tokenizer("fr")

        assert "Unsupported language ID 'fr'" in str(exc_info.value)

    @patch('spacy.load')  # Patch deeper
    def test_tokenize_spacy_success(self, mock_spacy_load):
        """Test with more real code execution"""
        # Mock what spaCy returns
        mock_sentences = [
            Mock(text="First sentence. "),
            Mock(text="Second sentence! "),
            Mock(text="Third sentence? ")
        ]
        mock_doc = Mock()
        mock_doc.sents = mock_sentences

        mock_nlp = Mock(return_value=mock_doc)
        mock_spacy_load.return_value = mock_nlp
        tokenizer = rr_utl.Tokenizer("en")
        result = tokenizer.tokenize("test")
        assert result == ("First sentence.", "Second sentence!", "Third sentence?")

    @patch('spacy.load')
    def test_tokenize_spacy_empty_result(self, mock_spacy_load):
        """Test with empty result"""
        mock_doc = Mock()
        mock_doc.sents = []
        mock_nlp = Mock(return_value=mock_doc)
        mock_spacy_load.return_value = mock_nlp

        # Executes SpaCyManager code too
        tokenizer = rr_utl.Tokenizer("en")
        result = tokenizer.tokenize("   ")

        assert result == ("   ",)
        mock_spacy_load.assert_called_with("en_core_web_sm")

    @patch('spacy.load')
    def test_tokenize_no_spacy_model(self, mock_spacy_load):
        """Test when spaCy model fails to load (returns None)"""
        # Simulate SpaCyManager returning None due to load failure
        mock_spacy_load.side_effect = OSError("Model not found")

        tokenizer = rr_utl.Tokenizer("en")
        result = tokenizer.tokenize("Hello. World! Test?")

        # Should use regex fallback
        assert result == ("Hello", "World", "Test")

    def test_tokenize_empty_text(self):
        """Test tokenization of empty text (uses regex fallback)"""
        # Mock SpaCyManager to return None to force regex fallback
        with patch.object(rr_utl.SpaCyManager, 'get_model', return_value=None):
            tokenizer = rr_utl.Tokenizer("en")

            # Regex handles these
            result = tokenizer.tokenize("")
            assert result == ()

            result = tokenizer.tokenize("   ")
            assert result == ()

            # Single word
            result = tokenizer.tokenize("hello")
            assert result == ("hello",)

    @patch('spacy.load')
    def test_tokenize_spacy_model_load_exception(self, mock_spacy_load):
        """Test spaCy load exception is handled and logs error"""
        mock_spacy_load.side_effect = OSError("Model 'en_core_web_sm' not found")

        # Mock SpaCyManager's logger instead of Tokenizer's
        with patch.object(rr_utl.SpaCyManager, 'logger') as mock_logger_method:
            mock_logger = Mock()
            mock_logger_method.return_value = mock_logger

            tokenizer = rr_utl.Tokenizer("en")
            result = tokenizer.tokenize("Test. Text.")

            # Should use regex fallback
            assert result == ("Test", "Text")
            # Verify error was logged by SpaCyManager
            mock_logger.error.assert_called()
            error_msg = mock_logger.error.call_args[0][0]
            assert "Failed to load spaCy model 'en'" in error_msg

    @patch('spacy.load')
    def test_tokenize_spacy_processing_exception(self, mock_spacy_load):
        """Test spaCy processing exception triggers regex fallback"""
        mock_nlp = Mock()
        mock_nlp.side_effect = Exception("spaCy processing error")
        mock_spacy_load.return_value = mock_nlp

        # Mock logger
        with patch.object(rr_utl.Tokenizer, 'logger') as mock_logger_method:
            mock_logger = Mock()
            mock_logger_method.return_value = mock_logger

            tokenizer = rr_utl.Tokenizer("en")
            result = tokenizer.tokenize("First. Second! Third?")

            # Should fall back to regex
            assert result == ("First", "Second", "Third")
            mock_logger.error.assert_called_once()
            assert "spaCy processing error" in mock_logger.error.call_args[0][0]

    @patch.object(rr_utl.SpaCyManager, 'get_model')
    def test_tokenize_regex_complex_punctuation(self, mock_get_model):
        """Test regex tokenization with complex English punctuation patterns"""
        mock_get_model.return_value = None  # Force regex fallback

        tokenizer = rr_utl.Tokenizer("en")

        # Test cases from original test
        text = "Wait... Really?! Oh no!!!"
        result = tokenizer.tokenize(text)
        assert result == ("Wait", "Really", "Oh no")

        # Additional complex cases
        text = "Hello... World??!! Test... End."
        result = tokenizer.tokenize(text)
        assert result == ("Hello", "World", "Test", "End")

        # Ellipsis with spaces
        text = "First ... Second ... Third"
        result = tokenizer.tokenize(text)
        assert result == ("First", "Second", "Third")

        # Mixed punctuation sequences
        text = "What?! No way... Seriously?!"
        result = tokenizer.tokenize(text)
        assert result == ("What", "No way", "Seriously")

        # Question-exclamation combos
        text = "Really?! Are you sure!? Impossible?!"
        result = tokenizer.tokenize(text)
        assert result == ("Really", "Are you sure", "Impossible")


class TestTokenizerIntegration:
    """
    INTEGRATION TESTS - NO MOCKING
    These will show actual coverage because they execute real code.
    Requires: pip install en_core_web_sm ja_core_news_sm
    """

    def setup_method(self):
        """Clear spaCy cache before each test"""
        rr_utl.SpaCyManager._nlp_mapping.clear()

    def test_tokenizer_init(self):
        # This executes: Tokenizer.__init__ -> SpaCyManager.get_model -> spacy.load
        tokenizer = rr_utl.Tokenizer("en")
        assert tokenizer.lang_id == "en"
        assert tokenizer._nlp is not None

        tokenizer = rr_utl.Tokenizer("jp")
        assert tokenizer.lang_id == "jp"
        assert tokenizer._nlp is not None

        with pytest.raises(ValueError) as exc_info:
            rr_utl.Tokenizer("fr")
        assert "Unsupported language ID 'fr'" in str(exc_info.value)

    def test_tokenize_spacy_success(self):
        """spaCy tokenization with actual NLP"""
        tokenizer = rr_utl.Tokenizer("en")

        result = tokenizer.tokenize("Hello world. This is a test!")
        assert isinstance(result, tuple)
        assert len(result) >= 1

    def test_tokenize_spacy_empty_text(self):
        """Empty text with spaCy"""
        tokenizer = rr_utl.Tokenizer("en")

        result = tokenizer.tokenize("")
        assert result == ("",)

        # Whitespace only
        result = tokenizer.tokenize("   ")
        assert result == ("   ",)

    def test_tokenize_spacy_single_sentence(self):
        """Single sentence testing"""
        tokenizer = rr_utl.Tokenizer("en")
        result = tokenizer.tokenize("Just one sentence")
        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_tokenize_regex_fallback_manual(self):
        """Test regex fallback by setting spaCy model to None"""
        # Force regex fallback by setting cache to None.
        with patch.object(rr_utl.SpaCyManager, '_nlp_mapping', {'en': None}):
            tokenizer = rr_utl.Tokenizer("en")
            result = tokenizer.tokenize("Hello. World! Test?")
            assert result == ("Hello", "World", "Test")

    def test_tokenize_complex_punctuation(self):
        """Complex punctuation with regex fallback testing"""
        # Force regex fallback
        with patch.object(rr_utl.SpaCyManager, '_nlp_mapping', {'en': None}):
            tokenizer = rr_utl.Tokenizer("en")

            text = "Wait... Really?! Oh no!!!"
            result = tokenizer.tokenize(text)
            assert result == ("Wait", "Really", "Oh no")

            text = "Hello... World??!! Test... End."
            result = tokenizer.tokenize(text)
            assert result == ("Hello", "World", "Test", "End")

    def test_tokenizer_property_lang_id(self):
        """REAL: Test lang_id property"""
        tokenizer = rr_utl.Tokenizer("en")
        assert tokenizer.lang_id == "en"

        # Property should be read-only
        with pytest.raises(AttributeError):
            tokenizer.lang_id = "jp"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
