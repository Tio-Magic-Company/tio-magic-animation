
from abc import abstractmethod, ABC
from typing import Any

class TextToVideoInterface(ABC):
    """Interface for text-to-video generation models"""
    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> Any:
        """Generate video from text prompt
        
        Args:
            prompt (str): The text prompt to generate video from
            **kwargs: Model-specific parameters
            
        Returns:
            str: Path to the generated video or video data
        """
        pass
