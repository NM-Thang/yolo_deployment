from typing import Any, Dict, List, Type

from core.base_strategy import AIStrategy

from strategies.fire_smoke import FireSmokeDetection
from strategies.intrusion import IntrusionDetection
from utils.detector_adapter import DetectorAdapter
from utils.triton_client import TritonClient

class AIEngine:
    """Manages the lifecycle and execution of AI strategies."""

    def __init__(self, ):
        # The Registry: Mapping string code to strategy classes        
        self._registry: Dict[str, Type[AIStrategy]] = {
            "fire_smoke": FireSmokeDetection,
            "intrusion": IntrusionDetection,
        }
        self.active_strategies: Dict[str, AIStrategy] = {}
        self.adapter = DetectorAdapter(TritonClient())

    def activate_by_type(self, strategies_config: dict) -> None:
        """Instantiate required AI classes based on code ."""
        for type_ai_code, config in strategies_config.items():
            if type_ai_code in self._registry and type_ai_code not in self.active_strategies:
                self.active_strategies[type_ai_code] = self._registry[type_ai_code](self.adapter, config)


    def analyze_frame(self, frame: Any) -> Dict[str, Any]:
        """Run the frame through all active AI models."""
        results = {}
        for code, strategy in self.active_strategies.items():
            results[code] = strategy.process_frame(frame)
        return results
