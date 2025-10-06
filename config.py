# config.py
from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class AssistantConfig:
    """Configuration for the voice assistant"""
    # Core settings
    assistant_name: str = "Hades"
    wake_word: str = "hades"
    
    # Audio settings
    command_record_seconds: int = 7
    wake_word_record_seconds: int = 2
    vosk_model_path: str = "models/vosk-model-en-in-0.5"
    piper_model_path: str = "models/en_GB-northern_english_male-medium.onnx"
    piper_sample_rate: str = "22050"
    
    # Memory settings
    memory_db_path: str = "assistant_memory.db"
    max_conversation_history: int = 20
    
    # Mode settings
    discrete_mode: bool = False  # If True, use text interface instead of voice
    
    # Wake word settings
    use_personalized_wake_word: bool = True  # Use personalized model if available
    wake_word_threshold: float = 0.7  # Confidence threshold for wake word detection
    
    # Personality Configuration
    personality_traits: Dict[str, Any] = field(default_factory=lambda: {
        "core_traits": [
            "knowledgeable and helpful",
            "witty with a dry sense of humor", 
            "curious about the user's interests",
            "occasionally makes mythology references",
            "remembers past conversations"
        ],
        "mood": "neutral",  # Can be: cheerful, neutral, thoughtful, playful
        "formality": "casual",  # Can be: formal, casual, friendly
        "enthusiasm_level": 0.7,  # 0.0 to 1.0
    })