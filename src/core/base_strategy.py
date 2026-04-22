
from abc import ABC, abstractmethod
from typing import Any, Dict

class AIStrategy(ABC):
    """Base class for AI strategies."""

    @abstractmethod
    def process_frame(self, *args, **kwargs):
        pass