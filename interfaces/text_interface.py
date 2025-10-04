from typing import Optional
import sys
from .base import AssistantInterface
from config import AssistantConfig

class TextInterface(AssistantInterface):
    """Text-based interface for discrete mode"""
    
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.first_interaction = True
        
    def initialize(self) -> None:
        """Initialize text interface"""
        print(f"\n💻 Text Interface Active - {self.config.assistant_name}")
        print("=" * 50)
        print("Type your messages below. Commands:")
        print(f"  - Type '{self.config.wake_word}' to start a conversation")
        print("  - Type 'exit', 'quit', or 'goodbye' to leave")
        print("  - Press Ctrl+C to force quit")
        print("=" * 50)
        
    def get_user_input(self, prompt: Optional[str] = None) -> Optional[str]:
        """Get text input from user"""
        try:
            if prompt == "wake_word":
                user_input = input("\n💭 > ").strip()
            else:
                user_input = input("\n💬 You: ").strip()
            
            return user_input if user_input else None
            
        except KeyboardInterrupt:
            print("\n\n⚡ Interrupted!")
            return "exit"
        except EOFError:
            return "exit"
    
    def output_response(self, text: str, emotion: str = "neutral") -> None:
        """Display text response"""
        if not text.strip():
            return
        
        # Emotional indicators
        emotion_indicators = {
            "happy": "😊",
            "cheerful": "😊",
            "thoughtful": "🤔",
            "playful": "😄",
            "neutral": "💭"
        }
        
        indicator = emotion_indicators.get(emotion, "💭")
        
        print(f"\n{indicator} {self.config.assistant_name}: {text}")
    
    def is_wake_word_detected(self, text: str) -> bool:
        """Check if wake word is in the text"""
        if not text:
            return False
        
        # In text mode, exact match or contained in message
        text_lower = text.lower().strip()
        return (text_lower == self.config.wake_word or 
                self.config.wake_word in text_lower)
    
    def cleanup(self) -> None:
        """Clean up text interface"""
        print("\n👋 Text interface shutting down...")