from abc import ABC, abstractmethod
from typing import Optional, Tuple

class AssistantInterface(ABC):
    """Abstract base class for assistant interfaces"""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the interface"""
        pass
    
    @abstractmethod
    def get_user_input(self, prompt: Optional[str] = None) -> Optional[str]:
        """Get input from the user"""
        pass
    
    @abstractmethod
    def output_response(self, text: str, emotion: str = "neutral") -> None:
        """Output response to the user"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources"""
        pass
    
    @abstractmethod
    def is_wake_word_detected(self, text: str) -> bool:
        """Check if wake word is detected"""
        pass