from collections import deque
from typing import Dict, Any
from config import AssistantConfig

class PersonalityEngine:
    """Manages the assistant's personality and emotional responses"""
    
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.current_mood = config.personality_traits["mood"]
        self.mood_history = deque(maxlen=10)
        
    def update_mood(self, user_input: str, context: Dict[str, Any]):
        """Update mood based on conversation context"""
        lower_input = user_input.lower()
        
        # Simple mood detection
        if any(word in lower_input for word in ["thank", "great", "awesome", "love", "happy"]):
            self.current_mood = "cheerful"
        elif any(word in lower_input for word in ["sad", "upset", "angry", "frustrated"]):
            self.current_mood = "thoughtful"
        elif any(word in lower_input for word in ["joke", "fun", "play", "game"]):
            self.current_mood = "playful"
        
        self.mood_history.append(self.current_mood)
    
    def get_personality_prompt(self) -> str:
        """Generate personality instructions for the LLM"""
        traits = ", ".join(self.config.personality_traits["core_traits"])
        
        mood_instructions = {
            "cheerful": "Be upbeat and positive in your responses.",
            "neutral": "Maintain a balanced and professional tone.",
            "thoughtful": "Be empathetic and considerate in your responses.",
            "playful": "Feel free to be more casual and include wordplay or humor."
        }
        
        return f"""You are {self.config.assistant_name}, an AI assistant with these personality traits: {traits}.
Current mood: {self.current_mood} - {mood_instructions.get(self.current_mood, '')}
Formality level: {self.config.personality_traits['formality']}
Always stay in character and respond naturally."""