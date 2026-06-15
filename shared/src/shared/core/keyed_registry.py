"""Generic keyed registry for grouping typed elements by a field value."""

from __future__ import annotations

from typing import Generic, Type, TypeVar

T = TypeVar("T")


class KeyedRegistry(Generic[T]):
    """Concrete generic registry that groups elements by a declared field value.

    Elements are stored in a dict of lists keyed by the value of key_field on
    each element. Can be used directly or subclassed to add domain-specific
    query methods on top of the base load / add / get / all operations.

    Usage:
        registry = KeyedRegistry(Rule, key_field="domain")
        registry.add(rule)
        registry.get_by_key("claim_appeal")

    Attributes:
        _registry: Internal store mapping key values to lists of elements.
    """

    def __init__(self, item_type: Type[T], key_field: str) -> None:
        """Initialize the registry.

        Args:
            item_type: The type of elements this registry holds.
            key_field: Name of the attribute on each element used as the grouping key.
        """
        self._item_type = item_type
        self._key_field = key_field
        self._registry: dict[str, list[T]] = {}

    def load(self, items: list[T]) -> None:
        """Replace registry contents with the provided elements.

        Clears the current state and re-groups all items by key_field.

        Args:
            items: Elements to load into the registry.
        """
        self._registry = {}
        for item in items:
            self._registry.setdefault(self._key(item), []).append(item)

    def add(self, item: T) -> None:
        """Append a single element to the registry.

        Args:
            item: Element to add.
        """
        self._registry.setdefault(self._key(item), []).append(item)

    def get_by_key(self, key: str) -> list[T]:
        """Return all elements stored under the given key.

        Args:
            key: Key value to look up, e.g. a domain name.

        Returns:
            List of elements for the key, or an empty list if not found.
        """
        return self._registry.get(key, [])

    def all(self) -> list[T]:
        """Return all elements across all keys as a flat list.

        Returns:
            Flat list of all elements in the registry.
        """
        return [item for items in self._registry.values() for item in items]

    def _key(self, item: T) -> str:
        """Extract the grouping key from an element.

        Args:
            item: Element to extract the key from.

        Returns:
            String value of key_field on the element.
        """
        return getattr(item, self._key_field)
