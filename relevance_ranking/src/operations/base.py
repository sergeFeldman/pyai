from abc import ABC, abstractmethod
from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, ClientError
import boto3
import hashlib
import json
from openai import OpenAI
import os
import pickle
from typing import Dict, List, Optional

import core as rr_core

# ========== MODULE-LEVEL SINGLETON CACHE ==========
_EMBED_CACHE = {}
_CACHE_MODIFIED = False
_CACHE_LOADED = False
_CACHE_FILE = "data/cache/embed.pkl"
# ==================================================


class ServiceClientManager(rr_core.BaseLoggable):
    """
    Utility static class, which instantiates and then catalogs AWS service(s).
    """

    _DEFAULT_REGION_NAME = "us-west-2"
    _service_clients_mapping: Dict[str, BaseClient] = {}

    @classmethod
    def service_client(cls, service_type: str, region_name: str = _DEFAULT_REGION_NAME) -> BaseClient:
        """
        Constructs and/or returns service client instance for given service type.

        Args:
            service_type (str): Type of service to create/return.
            region_name (str, default=_DEFAULT_REGION_NAME) Region name.

        Returns:
            Appropriate service client.
        """
        key = f"{service_type}-{region_name}"
        if key not in cls._service_clients_mapping:
            try:
                cls._service_clients_mapping[key] = boto3.client(service_type, region_name=region_name)
            except(BotoCoreError, ClientError) as e:
                error_msg = f"AWS {service_type} client creation error for in {region_name}: {str(e)}"
                cls.logger().error(error_msg)
                raise RuntimeError(error_msg) from e
            except Exception as e:
                error_msg = f"Unexpected error creating AWS {service_type} client in {region_name}: {str(e)}"
                cls.logger().error(error_msg)
                raise RuntimeError(error_msg) from e

        return cls._service_clients_mapping.get(key)


class ServiceClientManagerForOpenAI(rr_core.BaseLoggable):
    """
    Utility static class, which instantiates and manages OpenAI client.
    """
    _service_client = None

    @classmethod
    def service_client(cls) -> OpenAI:
        """Get or create OpenAI client."""
        if cls._service_client is None:
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                cls.logger().error("OPENAI_API_KEY environment variable not configured.")
                raise ValueError("OPENAI_API_KEY not configured")
            cls._service_client = OpenAI(api_key=api_key)
        return cls._service_client


class RelevanceScoreOperation(rr_core.BaseLoggable, ABC):
    """
    Abstract static class, which represents necessary logic
    to calculate relevant ranking score (both absolute and weighted) of
    provided text fragments.

    Note: Input validation is handled by calling components.
    Assumption is that valid, normalized (formatted) non-empty
    strings are provided during score calculation.
    """

    def __new__(cls, *args, **kwargs):
        raise TypeError(f"'{cls.__name__}' is an abstract class and cannot be instantiated.")

    @staticmethod
    @abstractmethod
    def calculate(text_fragment: str, text_comp_to_fragment: str) -> float:
        """
        Abstract static method, which calculates the raw relevance score of
        specific operation logic, which is implemented by sub-classes,
        using provided text fragments.

        Args:
            text_fragment (str): First text fragment, used in calculation.
            text_comp_to_fragment (str): Text fragment to compare to, used in calculation.

        Returns:
            float: Calculated value.
        """
        pass

    @classmethod
    def calculate_weighted(cls, text_fragment: str, text_comp_to_fragment: str, weight: float) -> float:
        """
        Calculate the normalized weighted score for provided text fragments,
        using given weight.

        Args:
            text_fragment (str): First text fragment, used in calculation.
            text_comp_to_fragment (str): Text fragment to compare to, used in calculation.
            weight (float): Weight of relevance score, used in calculation.

        Returns:
            float: Calculated value.
        """
        calculated_score = cls.calculate(text_fragment, text_comp_to_fragment)
        weighted_score = calculated_score * weight

        return weighted_score


class EmbeddableRelevanceScoreOperation(RelevanceScoreOperation):
    """
    Abstract class for operations requiring text embeddings with caching.
    Uses module-level singleton cache.
    """

    @classmethod
    def flush_cache(cls) -> None:
        """Explicitly write cache to disk."""
        cls._write_cache()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        global _CACHE_LOADED
        if not _CACHE_LOADED:
            cls._read_cache()
            _CACHE_LOADED = True

    @classmethod
    def _read_cache(cls) -> None:
        global _EMBED_CACHE
        if os.path.exists(_CACHE_FILE):
            try:
                with open(_CACHE_FILE, 'rb') as cache_file:
                    _EMBED_CACHE = pickle.load(cache_file)
            except Exception as e:
                cls.logger().warning(f"Cache read failed: {e}, starting fresh")
                _EMBED_CACHE = {}

    @classmethod
    def _write_cache(cls) -> None:
        global _CACHE_MODIFIED, _EMBED_CACHE
        if _CACHE_MODIFIED:
            try:
                os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
                with open(_CACHE_FILE, 'wb') as cache_file:
                    pickle.dump(_EMBED_CACHE, cache_file)
            except Exception as e:
                cls.logger().error(f"Cache write failed: {e}")

    @classmethod
    def embed(cls, input_string: str) -> Optional[List[float]]:
        """
        Encode provided string into embedding representation.

        Args:
            input_string (str): Input string to be encoded into embedding representation.

        Returns:
            Optional[List[float]]: Resulted embedding representation.
        """
        global _EMBED_CACHE, _CACHE_MODIFIED
        cache_key = hashlib.sha256(input_string.encode()).hexdigest()
        if cache_key not in _EMBED_CACHE:
            cls._populate_embed_cache(input_string, cache_key)
            _CACHE_MODIFIED = True
        return _EMBED_CACHE.get(cache_key)

    @classmethod
    @abstractmethod
    def _populate_embed_cache(cls, input_string: str, cache_key: str) -> None:
        """
        Service-specific implementation to populate cache.

        Args:
            input_string (str): Input string to be encoded into embedding representation.
            cache_key (str): Cache key to be used to store embedding representation.

        Returns:
            None
        """
        pass


class RelevanceScoreOperationWithBedrock(EmbeddableRelevanceScoreOperation, ServiceClientManager):
    """
    Abstract static class, which instantiates AWS Bedrock service(s).
    """

    _MODEL_ID = 'amazon.titan-embed-text-v1'

    @classmethod
    def _populate_embed_cache(cls, input_string: str, cache_key: str) -> None:
        """
        Generates encoding, using provided string and populates cache with its value.
        using model ID class attribute.

        Args:
            input_string (str): Input string to be encoded into embedding representation.
            cache_key (str): Cache key to be used to store embedding representation.

        Returns:
            None
        """
        global _EMBED_CACHE

        try:
            service_client = cls.service_client(service_type='bedrock-runtime')
            response = service_client.invoke_model(
                body=json.dumps({'inputText': input_string}),
                modelId=RelevanceScoreOperationWithBedrock._MODEL_ID,
                accept='application/json',
                contentType='application/json')

            response_body = json.loads(response.get('body').read())
            embedding = response_body.get('embedding')
            _EMBED_CACHE[cache_key] = embedding
        except Exception as e:
            cls.logger().error(f"Bedrock related embedding error for {input_string}: {str(e)}")
            raise


class RelevanceScoreOperationWithOpenAI(EmbeddableRelevanceScoreOperation, ServiceClientManagerForOpenAI):
    """
    Abstract static class for OpenAI embedding operations.
    """
    _MODEL_ID = 'text-embedding-3-small'

    @classmethod
    def _populate_embed_cache(cls, input_string: str, cache_key: str) -> None:
        """
        Generates embedding using OpenAI API and populates cache.

        Args:
            input_string (str): Input string to encode.
            cache_key (str): Cache key for storage.

        Returns:
            None
        """
        global _EMBED_CACHE

        try:
            service_client = cls.service_client()
            response = service_client.embeddings.create(model=cls._MODEL_ID, input=input_string)
            embedding = response.data[0].embedding
            _EMBED_CACHE[cache_key] = embedding
        except Exception as e:
            cls.logger().error(f"OpenAI related embedding error for {input_string}: {str(e)}")
            raise
