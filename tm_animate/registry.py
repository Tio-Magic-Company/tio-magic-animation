# decorator-based registration system
# maps model names to their corresponding model classes
# allows models to "self-register" by being imported
# keeping system clean and decoupled

from typing import Dict, Optional, Type, List, Any
from .adapters.base import AbstractAdapter
from .core import Feature

class _Registry:
    # singleton class
    def __init__(self) -> None:
        self._table: Dict[str, Type[AbstractAdapter]] = {}
        self._model_info: Dict[str, Dict[str, Any]] = {}

    def register(self, *model_types: str, **model_info: Any):
        # a decorator to register an adapter class with 1+ model names
        # @registry.register("model-v1, model-v2-alias", cost_per_second=0.01)
        # class MyModel(AbstractModel)
    
        def decorator(cclass: Type[AbstractAdapter]) -> Type[AbstractAdapter]:
            for name in model_types:
                if name in self._table:
                    raise KeyError(f"Model name '{name}' is already registered")
                self._table[name] = cclass
                self._model_info[name] = {
                    "adapter_class": cclass.__name__,
                    "supported features": [m.name for m in cclass.supported_features],
                    **model_info,
                }
            return cclass
        return decorator
    
    def get_adapter_class(self, name: str) -> Optional[Type[AbstractAdapter]]:
        """Retrieves the model class for a given model name."""
        return self._table.get(name)
    
    def list_models(self, modality: Optional[Feature] = None) -> List[str]:
        """Lists all registered model names, optionally filtering by features."""
        if modality is None:
            return sorted(list(self._table.keys()))
        
        return sorted([
            name for name, cls in self._table.items()
            if modality in cls.supported_modalities
        ])
    
    def get_model_info(self, name: str) -> Dict[str, Any]:
        """Retrieves the metadata for a specific model."""
        if name not in self._model_info:
            raise KeyError(f"No information found for model '{name}'.")
        return self._model_info[name]

# Create a single, global instance to be used throughout the library.
registry = _Registry()
