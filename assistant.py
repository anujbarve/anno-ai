# assistant.py
"""
Core assistant logic without LLM dependency
You'll need to implement your own LLM integration
"""

from typing import Optional
from config import AssistantConfig
from memory import ConversationMemory
from personality import PersonalityEngine
from features import FeatureManager
from utils import extract_information, clean_response
from interfaces import VoiceInterface, TextInterface

class EnhancedAssistant:
    """Enhanced assistant with personality and memory"""
    
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.memory = ConversationMemory(
            config.memory_db_path,
            config.max_conversation_history
        )
        self.personality = PersonalityEngine(config)
        self.features = FeatureManager(self.memory)
        
        # Choose interface based on mode
        if config.discrete_mode:
            self.interface = TextInterface(config)
        else:
            self.interface = VoiceInterface(config)
        
        self.is_active = False
        self.conversation_active = False
    
    def initialize(self):
        """Initialize the assistant"""
        self.interface.initialize()
        self.is_active = True
    
    def run(self):
        """Main assistant loop"""
        while self.is_active:
            try:
                if not self.conversation_active:
                    # Wait for wake word
                    user_input = self.interface.get_user_input(prompt="wake_word")
                    
                    if not user_input:
                        continue
                    
                    if self.interface.is_wake_word_detected(user_input):
                        self.conversation_active = True
                        self.interface.output_response(
                            f"Yes? How can I help you?",
                            emotion="cheerful"
                        )
                    continue
                
                # Get user command
                user_input = self.interface.get_user_input()
                
                if not user_input:
                    continue
                
                # Check for exit commands
                if self._is_exit_command(user_input):
                    self.interface.output_response(
                        "Goodbye! Have a great day!",
                        emotion="cheerful"
                    )
                    self.is_active = False
                    break
                
                # Process the input
                response, emotion = self._process_input(user_input)
                
                # Output response
                if response:
                    self.interface.output_response(response, emotion)
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                continue
    
    def _process_input(self, user_input: str) -> tuple[str, str]:
        """
        Process user input and generate response
        
        TODO: Integrate your LLM of choice here
        This is a placeholder that returns basic responses
        """
        # Extract information
        extracted = extract_information(user_input)
        
        # Save user info if found
        for key, value in extracted.get("user_info", {}).items():
            self.memory.save_user_info(key, value)
        
        # Update personality mood
        self.personality.update_mood(user_input, extracted)
        
        # TODO: Replace this with your LLM call
        # This is just a placeholder
        response = self._generate_placeholder_response(user_input, extracted)
        
        # Clean and save
        response = clean_response(response)
        emotion = self.personality.current_mood
        
        # Save conversation
        self.memory.add_conversation(
            user_input,
            response,
            emotion,
            extracted.get("topics", [])
        )
        
        return response, emotion
    
    def _generate_placeholder_response(self, user_input: str, extracted: dict) -> str:
        """
        Placeholder response generator
        
        TODO: Replace this with your actual LLM integration
        """
        user_input_lower = user_input.lower()
        
        # Basic pattern matching responses
        if "hello" in user_input_lower or "hi" in user_input_lower:
            return "Hello! How can I assist you today?"
        
        elif "how are you" in user_input_lower:
            return "I'm functioning well, thank you for asking! How can I help you?"
        
        elif "name" in user_input_lower and "your" in user_input_lower:
            return f"My name is {self.config.assistant_name}."
        
        elif "thank" in user_input_lower:
            return "You're welcome! Is there anything else I can help you with?"
        
        elif "joke" in user_input_lower:
            return "Why don't scientists trust atoms? Because they make up everything!"
        
        else:
            return ("I understand you said: " + user_input + 
                   ". However, I need to be connected to an LLM to provide meaningful responses. "
                   "Please integrate your preferred LLM in the _process_input method.")
    
    def _is_exit_command(self, text: str) -> bool:
        """Check if user wants to exit"""
        exit_words = ["goodbye", "bye", "exit", "quit", "stop"]
        return any(word in text.lower() for word in exit_words)
    
    def cleanup(self):
        """Clean up resources"""
        self.interface.cleanup()