"""Service registry for the GuardGPT MCP server.

Every callable the server exposes is described here once: its name, which layer
of the architecture it belongs to, whether it is the canonical end-to-end path,
and a handler that accepts a plain argument mapping and returns a plain mapping.

The registry is the single place that knows what the server can do, so the
server itself only has to wire typed MCP signatures onto these entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

__all__ = [
    "ServiceSpec",
    "ServiceRegistry",
    "ServiceNotFound",
    "DuplicateService",
]

Handler = Callable[[dict], dict]


class ServiceNotFound(KeyError):
    """Raised when a requested service is not registered."""

    def __str__(self) -> str:  # KeyError quotes its message by default
        return str(self.args[0]) if self.args else "unknown service"


class DuplicateService(ValueError):
    """Raised when a service name is registered twice."""


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    layer: str
    description: str
    handler: Handler
    canonical: bool = False
    deprecated: bool = False
    # False for report adapters that stop at the input checks. Only a service
    # that consults the Jev-Style verdict may call itself authoritative.
    authoritative: bool = True

    def describe(self) -> dict:
        return {
            "name": self.name,
            "layer": self.layer,
            "canonical": self.canonical,
            "deprecated": self.deprecated,
            "authoritative": self.authoritative,
            "description": self.description,
        }


class ServiceRegistry:
    """Name -> :class:`ServiceSpec` mapping with duplicate protection."""

    def __init__(self) -> None:
        self._services: dict[str, ServiceSpec] = {}

    def register(self, spec: ServiceSpec) -> ServiceSpec:
        if spec.name in self._services:
            raise DuplicateService(f"{spec.name} is already registered")
        self._services[spec.name] = spec
        return spec

    def get(self, name: str) -> ServiceSpec:
        try:
            return self._services[name]
        except KeyError:
            raise ServiceNotFound(f"Unknown service: {name}") from None

    def names(self) -> tuple[str, ...]:
        return tuple(self._services)

    def services(self) -> tuple[ServiceSpec, ...]:
        return tuple(self._services.values())

    def describe(self) -> list[dict]:
        return [spec.describe() for spec in self._services.values()]

    def __len__(self) -> int:
        return len(self._services)

    def __contains__(self, name: object) -> bool:
        return name in self._services
