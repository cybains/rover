from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List
from ..schemas import NormalizedDoc




class Connector(ABC):
name: str = "base"


@abstractmethod
def fetch(self, **params) -> Iterable[Dict[str, Any]]:
    ...


@abstractmethod
def normalize(self, payload: Dict[str, Any]) -> NormalizedDoc:
    ...


def run(self, **params) -> List[NormalizedDoc]:
    return [self.normalize(p) for p in self.fetch(**params)]