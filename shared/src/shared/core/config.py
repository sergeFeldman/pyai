"""Configurable related classes"""

from abc import ABC
import json
import logging
from typing import Generic, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

from .singleton import Singleton

T = TypeVar('T')

PYDANTIC_TYPE_ERROR_TYPES = {
    "bool_type", "bytes_type", "dict_type", "float_type", "int_type", "list_type", 
    "model_type", "set_type", "string_type", "tuple_type"
}

logger = logging.getLogger(__name__)


class Configurable(ABC, Generic[T]):
    """Abstract base class for objects initialized from validated config data.

    Concrete subclasses must declare the `_config_data_type` class attribute.
    That type is used by the associated factory to validate and convert an
    input dictionary into a strongly typed configuration object.
    """

    # Configuration model associated with the concrete subclass.
    # The factory uses this type to validate and convert input data before
    # constructing the configurable object.
    _config_data_type: Type[T]

    @classmethod
    def config_data_type(cls) -> Type[T]:
        """Return the configuration type declared by the concrete subclass.

        Returns:
            Type[T]: Configuration type associated with the subclass.
        """
        return cls._config_data_type

    def __init__(self, config: T):
        """Initialize the configurable object.

        Args:
            config (T): Validated configuration object associated with the
                concrete configurable subtype.

        Raises:
            ValueError: Raised if the provided config is None.
        """
        if config is None:
            raise ValueError("Config parameter cannot be None")
        self._config = config

    @property
    def config(self) -> T:
        """Return the validated configuration object set during initialization.

        Returns:
            T: Configuration object associated with the instance.
        """
        return self._config


class ConfigurableObjectFactory(metaclass=Singleton):
    """Abstract singleton factory class for creating Configurable objects.

    This implementation also contains conversion logic for transforming
    a provided config mapping into a validated configuration model.

    The cls._TYPES_MAPPING protected class mapping is a constant structure
    containing static type reference definitions. It must be assigned
    during subclass factory implementation and cannot be None.

    The self._objects protected instance dictionary contains all transient
    objects created within the current process utilization of the factory
    subclass.
    """

    _TYPES_MAPPING = None

    def __init__(self, comprehensive_hashing=False):
        """Initialize the factory.

        Args:
            comprehensive_hashing (bool; default=False): Hashing mechanism used
                by the current factory class when instantiating and storing
                objects. If False, a simple mechanism based on the provided
                identifier is used. If True, a more complex one is used by
                concatenating the provided identifier and the hash of the
                provided structure.

        Raises:
            ValueError: Raised when the static _TYPES_MAPPING container is
                incorrectly initialized or empty.
        """

        if self._TYPES_MAPPING is None or len(self._TYPES_MAPPING) == 0:
            raise ValueError(f"Given Factory implementation {self.__class__.__name__} is invalid, \
                             since TYPES_MAPPING is not properly assigned")
        
        # Initialize the object container.
        self._objects = {}
        self._comprehensive_hashing = comprehensive_hashing

    @staticmethod
    def convert_from_dict(data_class_type: Type[T], input_data: dict, strict_convert=False) -> Union[T, dict]:
        """Conversion routine between provided dictionary and a target type.

        How validation and verification works during the conversion process:
        - If the target type is dict, the input dictionary is returned as-is.
        - If the target type is a Pydantic model, input data is validated by
          using the model validation routine.
        - Missing required attributes raise KeyError.
        - Unexpected input attributes are ignored by default.
        - Above behavior can be altered by setting strict_convert to True. In
          that case, unexpected keys are preserved and are expected to be
          rejected by a strict Pydantic model configuration.
        - Wrong input value types raise TypeError.

        Args:
            data_class_type (Type[T]): Target type of configuration model.
            input_data (dict): Provided data dictionary.
            strict_convert (bool; default=False): Validation/verification method indicator,
                used during conversion.

        Returns:
            Union[T, dict]: Validated target object.

        Raises:
            KeyError: Raised when a required field is missing.
            KeyError: Raised when strict mode is enabled and the input data contains unexpected keys.
            RuntimeError: Raised in case of all other erroneous cases.
            TypeError: Raised when an input value type does not match with target field type.
        """

        try:
            if data_class_type == dict:
                return input_data

            if not issubclass(data_class_type, BaseModel):
                raise RuntimeError(f"Unsupported configuration type {data_class_type}; "
                                   "expected dict or Pydantic BaseModel subclass")

            model_data = (input_data if strict_convert 
                          else {key: value for key, value in input_data.items() 
                                if key in data_class_type.model_fields})
            
            return data_class_type.model_validate(model_data)
        except ValidationError as e:
            error_types = {error["type"] for error in e.errors()}

            if "missing" in error_types:
                missing_fields = [
                    ".".join(str(part) for part in error["loc"])
                    for error in e.errors() if error["type"] == "missing"
                ]
                raise KeyError(f"Mandatory attribute(s) {missing_fields} missing for "
                               f"target config model {data_class_type}") from e

            if "extra_forbidden" in error_types:
                unexpected_keys = [
                    ".".join(str(part) for part in error["loc"])
                    for error in e.errors() if error["type"] == "extra_forbidden"
                ]
                raise KeyError(f"Provided input data container has unexpected keys "
                               f"{unexpected_keys} in strict mode") from e

            if error_types.issubset(PYDANTIC_TYPE_ERROR_TYPES):
                raise TypeError(f"Wrong value type provided for target config model "
                                f"{data_class_type}: {e}") from e

            raise RuntimeError(f"Validation error converting data into {data_class_type}: "
                               f"{e}") from e
        except Exception as e:
            raise RuntimeError(f"Error converting data into {data_class_type} with the "
                               f"following error: {e}") from e
    
    def get_obj(self, _id: str, config: dict, replace_with_new: bool = False) -> Configurable:
        """Extract the object instance based on the provided identifier.

        Args:
            _id (str): Identifier.
            config (dict): Provided configuration container.
            replace_with_new (bool, default=False): Replacement indicator.

        Returns:
            Configurable: Concrete instance of an object.
        """

        # Default behavior of hash code generation.
        hash_code = self._generate_hash(_id, config) if self._comprehensive_hashing else _id
        if hash_code not in self._objects or replace_with_new:
            self._objects[hash_code] = self._create_obj(_id, config)
            logger.info(f"{type(self._objects[hash_code]).__name__} instance created.")

        logger.info(f"{type(self._objects[hash_code]).__name__} instance returned.")
        return self._objects[hash_code]
    
    async def get_obj_async(self, _id: str, config: dict, replace_with_new: bool = False) -> Configurable:
        """Extract the object instance asynchronously based on the provided identifier.

        Args:
            _id (str): Identifier.
            config (dict): Provided configuration container.
            replace_with_new (bool, default=False): Replacement indicator.

        Returns:
            Configurable: Concrete instance of an object.
        """
        hash_code = self._generate_hash(_id, config) if self._comprehensive_hashing else _id
        if hash_code not in self._objects or replace_with_new:
            self._objects[hash_code] = await self._create_obj_async(_id, config)
            logger.info(f"{type(self._objects[hash_code]).__name__} instance created.")

        logger.info(f"{type(self._objects[hash_code]).__name__} instance returned.")
        return self._objects[hash_code]

    async def _create_obj_async(self, _id: str, config: dict) -> Configurable:
        """Create the object instance asynchronously. Falls back to sync by default.

        Override in subclasses that require async instantiation.

        This method is for internal use only.

        Args:
            _id (str): Identifier.
            config (dict): Provided configuration container.

        Returns:
            Configurable: Concrete instance of an object.
        """
        return self._create_obj(_id, config)

    def _create_obj(self, _id: str, config: dict) -> Configurable:
        """Create the object instance based on the provided identifier.

        Also convert the provided configuration container to the necessary
        config type of the Configurable class.

        Default functionality converts an input dictionary into a validated
        configuration model associated with the current Configurable subclass.
        If a different behavior is needed, this method must be overridden.

        This method is for internal use only.

        Args:
            _id (str): Identifier.
            config (dict): Provided configuration container.

        Returns:
            Configurable: Concrete instance of an object.
        """

        if _id not in self._TYPES_MAPPING:
            raise ValueError(f"Provided identifier is not supported: {_id}")
        
        configurable_type = self._TYPES_MAPPING[_id]
        target_config = self.convert_from_dict(configurable_type.config_data_type(), config)
        return configurable_type(target_config)
    
    @staticmethod
    def _generate_hash(input_id: str, dataset:dict) -> str:
        """Generate a hash-based unique key string.

        The key is created by concatenating input_id and hash(dataset).

        This method is for internal use only.

        Args:
            input_id (str): Provided identifier.
            dataset (dict): Provided dataset container.

        Returns:
            target_key (str): Generated key.
        """

        return input_id + str(hash(json.dumps(dataset, sort_keys=True, default=str)))
    
