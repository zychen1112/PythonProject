"""State serialization utilities."""

import json
from datetime import datetime
from typing import Any, Callable, Optional


class StateSerializer:
    """Utilities for serializing and deserializing state.

    Supports JSON, and provides extensibility for custom types.
    """

    # Custom encoders for specific types
    _custom_encoders: dict[type, Callable[[Any], Any]] = {}
    _custom_decoders: dict[str, Callable[[Any], Any]] = {}

    @classmethod
    def register_encoder(
        cls,
        type_: type,
        encoder: Callable[[Any], Any],
        decoder: Optional[Callable[[Any], Any]] = None,
        type_name: Optional[str] = None,
    ) -> None:
        """Register a custom encoder/decoder for a type.

        Args:
            type_: The type to encode
            encoder: Function to convert type to JSON-serializable
            decoder: Function to convert back (optional)
            type_name: Type name for decoder lookup (defaults to class name)
        """
        cls._custom_encoders[type_] = encoder
        if decoder:
            type_name = type_name or type_.__name__
            cls._custom_decoders[type_name] = decoder

    @classmethod
    def serialize(cls, state: dict[str, Any], format: str = "json") -> bytes:
        """Serialize state to bytes.

        Args:
            state: State dictionary to serialize
            format: Serialization format (json)

        Returns:
            Serialized bytes
        """
        if format != "json":
            raise ValueError(f"Unsupported format: {format}")

        # Apply custom encoders
        encoded_state = cls._encode_custom(state)

        return json.dumps(
            encoded_state,
            default=cls._json_default,
            ensure_ascii=False,
        ).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes, format: str = "json") -> dict[str, Any]:
        """Deserialize bytes to state.

        Args:
            data: Serialized bytes
            format: Serialization format (json)

        Returns:
            State dictionary
        """
        if format != "json":
            raise ValueError(f"Unsupported format: {format}")

        state = json.loads(data.decode("utf-8"))

        # Apply custom decoders
        return cls._decode_custom(state)

    @classmethod
    def _encode_custom(cls, obj: Any) -> Any:
        """Encode custom types in an object."""
        if isinstance(obj, dict):
            return {k: cls._encode_custom(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [cls._encode_custom(item) for item in obj]
        elif type(obj) in cls._custom_encoders:
            encoded = cls._custom_encoders[type(obj)](obj)
            encoded["__type__"] = type(obj).__name__
            return encoded
        return obj

    @classmethod
    def _decode_custom(cls, obj: Any) -> Any:
        """Decode custom types in an object."""
        if isinstance(obj, dict):
            if "__type__" in obj:
                type_name = obj.pop("__type__")
                if type_name in cls._custom_decoders:
                    return cls._custom_decoders[type_name](obj)
                # Return as-is if no decoder
                obj["__type__"] = type_name
            return {k: cls._decode_custom(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [cls._decode_custom(item) for item in obj]
        return obj

    @staticmethod
    def _json_default(obj: Any) -> Any:
        """Default JSON encoder for common types."""
        if isinstance(obj, datetime):
            return {"__type__": "datetime", "value": obj.isoformat()}
        elif isinstance(obj, set):
            return {"__type__": "set", "value": list(obj)}
        elif hasattr(obj, "__dict__"):
            return {"__type__": type(obj).__name__, "value": obj.__dict__}
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# Register built-in type handlers
StateSerializer._custom_decoders["datetime"] = lambda d: datetime.fromisoformat(d["value"])
StateSerializer._custom_decoders["set"] = lambda d: set(d["value"])


def serialize_state(state: dict[str, Any]) -> bytes:
    """Convenience function to serialize state.

    Args:
        state: State to serialize

    Returns:
        Serialized bytes
    """
    return StateSerializer.serialize(state)


def deserialize_state(data: bytes) -> dict[str, Any]:
    """Convenience function to deserialize state.

    Args:
        data: Serialized bytes

    Returns:
        State dictionary
    """
    return StateSerializer.deserialize(data)
