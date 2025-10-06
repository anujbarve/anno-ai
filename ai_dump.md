from .base import AssistantInterface
from .voice_interface import VoiceInterface
from .text_interface import TextInterface

__all__ = ['AssistantInterface', 'VoiceInterface', 'TextInterface']

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

import os
import struct
import json
import subprocess
import numpy as np
from typing import Optional
from vosk import Model, KaldiRecognizer
from pvrecorder import PvRecorder
from .base import AssistantInterface
from config import AssistantConfig

# Try to import personalized wake word detector
try:
    from voice_training import PersonalizedWakeWordDetector
    PERSONALIZED_AVAILABLE = True
except ImportError:
    PERSONALIZED_AVAILABLE = False

class VoiceInterface(AssistantInterface):
    """Voice-based interface using STT and TTS"""
    
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.model = None
        self.recognizer = None
        self.recorder = None
        self.personalized_detector = None
        
        # Check for personalized wake word model
        self.personalized_model_path = f"models/wake_word_{config.wake_word}.pkl"
        
    def initialize(self) -> None:
        """Initialize voice components"""
        if not os.path.exists(self.config.vosk_model_path):
            raise FileNotFoundError(f"Vosk model not found at '{self.config.vosk_model_path}'")
        
        if not os.path.exists(self.config.piper_model_path):
            raise FileNotFoundError(f"Piper model not found at '{self.config.piper_model_path}'")
        
        self.model = Model(self.config.vosk_model_path)
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.recorder = PvRecorder(device_index=-1, frame_length=512)
        
        # Try to load personalized wake word model
        if PERSONALIZED_AVAILABLE and os.path.exists(self.personalized_model_path):
            try:
                print("🎯 Loading personalized wake word model...")
                self.personalized_detector = PersonalizedWakeWordDetector(self.personalized_model_path)
                print("✅ Personalized wake word detection enabled!")
            except Exception as e:
                print(f"⚠️  Could not load personalized model: {e}")
                print("   Falling back to standard wake word detection")
    
    def get_user_input(self, prompt: Optional[str] = None) -> Optional[str]:
        """Get voice input from user"""
        if prompt == "wake_word":
            # Use personalized detection if available
            if self.personalized_detector:
                return self._detect_wake_word_personalized()
            else:
                return self._transcribe_audio(self.config.wake_word_record_seconds)
        else:
            return self._transcribe_audio(self.config.command_record_seconds)
    
    def _detect_wake_word_personalized(self) -> Optional[str]:
        """Detect wake word using personalized model"""
        try:
            self.recorder.start()
            print(f"🎤 Listening for personalized wake word...")
            
            # Record audio
            frames = []
            num_frames = int((self.recorder.sample_rate / self.recorder.frame_length) * self.config.wake_word_record_seconds)
            
            for i in range(num_frames):
                frames.extend(self.recorder.read())
            
            self.recorder.stop()
            
            # Convert to numpy array
            audio_data = np.array(frames, dtype=np.float32) / 32768.0
            
            # Check with personalized model
            is_wake_word, confidence = self.personalized_detector.is_wake_word(audio_data)
            
            if is_wake_word:
                print(f"✅ Personalized wake word detected! (confidence: {confidence:.2%})")
                return self.config.wake_word
            else:
                # Also try with Vosk for fallback
                frame_bytes = struct.pack("h" * len(frames), *frames)
                if self.recognizer.AcceptWaveform(frame_bytes):
                    result = self.recognizer.Result()
                else:
                    result = self.recognizer.PartialResult()
                
                text = json.loads(result).get("text", "").lower().strip()
                
                # If Vosk detected the wake word but personalized didn't, log it
                if text and self.config.wake_word in text:
                    print(f"⚠️  Vosk detected wake word but personalized model didn't (confidence: {confidence:.2%})")
                
                return text if text else None
                
        except Exception as e:
            print(f"Error in personalized wake word detection: {e}")
            # Fall back to standard detection
            return self._transcribe_audio(self.config.wake_word_record_seconds)
    
    def output_response(self, text: str, emotion: str = "neutral") -> None:
        """Speak the response using TTS"""
        if not text.strip():
            return
        
        # Emotional prefixes for display
        emotional_prefix = {
            "happy": "😊 ",
            "cheerful": "😊 ",
            "thoughtful": "🤔 ",
            "playful": "😄 ",
            "neutral": ""
        }
        
        # Clean text for shell command
        cleaned_text = (text.replace('"', "'")
                           .replace("`", "'")
                           .replace("$", "")
                           .replace("\\", "")
                           .replace("\n", " ")
                           .strip())
        
        prefix = emotional_prefix.get(emotion, "")
        print(f"\n{prefix}{self.config.assistant_name}: {cleaned_text}")
        
        command = (
            f'echo "{cleaned_text}" | '
            f'flatpak-spawn --host piper --model {self.config.piper_model_path} --output-raw | '
            f'flatpak-spawn --host aplay -r {self.config.piper_sample_rate} -f S16_LE -t raw -'
        )
        
        try:
            subprocess.run(
                command,
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError as e:
            print(f"Error running Piper TTS: {e}")
        except FileNotFoundError:
            print("Error: A command in the pipeline was not found.")
    
    def _transcribe_audio(self, duration: int) -> Optional[str]:
        """Transcribe audio to text"""
        try:
            self.recorder.start()
            print(f"🎤 Listening for {duration} seconds...")
            
            frames = []
            num_frames = int((self.recorder.sample_rate / self.recorder.frame_length) * duration)
            
            for i in range(num_frames):
                frames.extend(self.recorder.read())
                
                # Progress indicator
                if i % (num_frames // 10) == 0:
                    progress = int((i / num_frames) * 10)
                    print("█" * progress + "░" * (10 - progress), end='\r')
            
            print("\n✅ Processing audio...")
            self.recorder.stop()
            
            # Convert to bytes
            frame_bytes = struct.pack("h" * len(frames), *frames)
            
            # Transcribe
            if self.recognizer.AcceptWaveform(frame_bytes):
                result = self.recognizer.Result()
            else:
                result = self.recognizer.PartialResult()
            
            text = json.loads(result).get("text", "").strip()
            
            if text:
                print(f"📝 Heard: \"{text}\"")
                
            return text if text else None
            
        except Exception as e:
            print(f"❌ Error in transcription: {e}")
            return None
    
    def is_wake_word_detected(self, text: str) -> bool:
        """Check if wake word is in the text"""
        # For personalized detection, this is handled differently
        # This method is mainly for the text returned by transcription
        return text and self.config.wake_word in text.lower()
    
    def cleanup(self) -> None:
        """Clean up voice resources"""
        if self.recorder:
            self.recorder.delete()
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
    
    # AI settings
    ollama_model: str = "qwen3:0.6b"
    
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
import random
from memory import ConversationMemory

class FeatureManager:
    """Manages additional features and capabilities"""
    
    def __init__(self, memory: ConversationMemory):
        self.memory = memory
        self.active_timers = {}
        
    def set_reminder(self, content: str, time_str: str) -> str:
        """Set a reminder (simplified version)"""
        self.memory.add_memory(
            f"Reminder: {content} at {time_str}",
            importance=0.8,
            category="reminder"
        )
        return f"I'll remind you to {content} at {time_str}."
    
    def get_fun_fact(self) -> str:
        """Return a random fun fact"""
        facts = [
            "Did you know? The term 'computer bug' originated when a real moth was found in a Harvard computer in 1947.",
            "Fun fact: Honey never spoils. Archaeologists have found 3000-year-old honey that's still edible!",
            "Here's something interesting: Octopi have three hearts and blue blood.",
            "Did you know? The first computer programmer was Ada Lovelace in the 1840s.",
            "Mythology fact: In Greek mythology, Hades' three-headed dog Cerberus guards the underworld.",
            "The longest word you can type using only the left hand is 'stewardesses'.",
            "A group of flamingos is called a 'flamboyance'.",
            "The first oranges weren't orange - they were green!"
        ]
        return random.choice(facts)
#!/usr/bin/env python3
"""
Enhanced Voice Assistant - Main Entry Point
Supports both voice and text interfaces
"""

import os
import sys
import argparse
import ollama
from typing import Optional
from config import AssistantConfig
from assistant import EnhancedAssistant
from memory import ConversationMemory
from personality import PersonalityEngine

def check_requirements(config: AssistantConfig) -> bool:
    """Check if required files and services are available"""
    print("🔍 Checking requirements...")
    
    all_good = True
    
    # Check models only if not in discrete mode
    if not config.discrete_mode:
        required_files = {
            "Vosk Model": config.vosk_model_path,
            "Piper Model": config.piper_model_path
        }
        
        for name, path in required_files.items():
            if os.path.exists(path):
                print(f"✅ {name}: Found")
            else:
                print(f"❌ {name}: Missing at {path}")
                all_good = False
    else:
        print("✅ Discrete mode: Skipping voice model checks")
    
    # Check Ollama connection
    print("\n🔌 Checking Ollama connection...")
    try:
        test_response = ollama.chat(
            model=config.ollama_model,
            messages=[{'role': 'user', 'content': 'test'}],
            stream=False
        )
        print("✅ Ollama: Connected")
    except Exception as e:
        print(f"❌ Ollama: Failed to connect - {e}")
        print("Please make sure Ollama is running and the model is installed.")
        all_good = False
    
    return all_good

def create_custom_assistant(args: argparse.Namespace) -> Optional[EnhancedAssistant]:
    """Create assistant with custom configuration"""
    config = AssistantConfig()
    
    # Apply command line arguments
    if args.name:
        config.assistant_name = args.name
        config.wake_word = args.name.lower()
    
    if args.wake_word:
        config.wake_word = args.wake_word.lower()
    
    if args.discrete:
        config.discrete_mode = True
    
    if args.model:
        config.ollama_model = args.model
    
    if args.memory_db:
        config.memory_db_path = args.memory_db
    
    # Check requirements
    if not check_requirements(config):
        return None
    
    return EnhancedAssistant(config)

def test_memory_system(db_path: str = "test_memory.db"):
    """Test the memory system independently"""
    print("🧪 Testing memory system...")
    memory = ConversationMemory(db_path)
    
    # Test saving and retrieving user info
    memory.save_user_info("name", "TestUser", "personal")
    memory.save_user_info("age", "25", "personal")
    memory.save_user_info("likes_1", "pizza", "preferences")
    
    info = memory.get_user_info()
    print(f"Stored user info: {info}")
    
    # Test conversation history
    memory.add_conversation("Hello", "Hi there!", "cheerful", ["greeting"])
    memory.add_conversation("What's the weather?", "I don't have weather data.", "neutral", ["weather"])
    
    history = memory.get_recent_conversations(5)
    print(f"Conversation history: {history}")
    
    print("✅ Memory system test complete!")

def test_personality_engine():
    """Test the personality engine"""
    print("🧪 Testing personality engine...")
    config = AssistantConfig()
    personality = PersonalityEngine(config)
    
    test_inputs = [
        "Thank you so much!",
        "I'm feeling sad today",
        "Tell me a joke",
        "What's the weather?"
    ]
    
    for input_text in test_inputs:
        personality.update_mood(input_text, {})
        print(f"Input: '{input_text}' -> Mood: {personality.current_mood}")
    
    print("✅ Personality engine test complete!")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced Voice Assistant with personality and memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with voice interface (default)
  python main.py
  
  # Run in discrete/text mode
  python main.py --discrete
  
  # Run with custom name and wake word
  python main.py --name "Jarvis" --wake-word "jarvis"
  
  # Run tests
  python main.py --test-memory
  python main.py --test-personality
        """
    )
    
    # Mode options
    parser.add_argument('--discrete', '-d', action='store_true',
                      help='Run in discrete mode (text only, no TTS/STT)')
    
    # Customization options
    parser.add_argument('--name', type=str,
                      help='Assistant name (default: Hades)')
    parser.add_argument('--wake-word', type=str,
                      help='Wake word to activate assistant')
    parser.add_argument('--model', type=str,
                      help='Ollama model to use')
    parser.add_argument('--memory-db', type=str,
                      help='Path to memory database file')
    
    # Test options
    parser.add_argument('--test-memory', action='store_true',
                      help='Test the memory system')
    parser.add_argument('--test-personality', action='store_true',
                      help='Test the personality engine')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run tests if requested
    if args.test_memory:
        test_memory_system()
        return
    
    if args.test_personality:
        test_personality_engine()
        return
    
    # Print banner
    mode = "Discrete/Text" if args.discrete else "Voice"
    print(f"""
    ╔══════════════════════════════════════════╗
    ║      Enhanced Voice Assistant v2.0       ║
    ║         Powered by Local AI              ║
    ║          Mode: {mode:^20} ║
    ╚══════════════════════════════════════════╝
    """)
    
    # Create and run assistant
    try:
        assistant = create_custom_assistant(args)
        if assistant:
            print("\n🚀 Initializing assistant...")
            assistant.initialize()
            
            print("✨ Assistant ready!\n")
            if args.discrete:
                print("💬 Type your messages to interact")
            else:
                print(f"🎤 Say '{assistant.config.wake_word}' to wake me")
            print("🛑 Say 'goodbye' or press Ctrl+C to exit\n")
            
            assistant.run()
        else:
            print("\n❌ Failed to initialize assistant. Please check the errors above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚡ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

import sqlite3
import json
import datetime
from typing import List, Dict, Any, Optional
from collections import deque

class ConversationMemory:
    """Manages persistent memory and conversation history"""
    
    def __init__(self, db_path: str, max_history: int = 20):
        self.db_path = db_path
        self.max_history = max_history
        self.short_term_memory = deque(maxlen=10)
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for memory storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Conversation history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_input TEXT,
                assistant_response TEXT,
                emotion TEXT,
                topics TEXT
            )
        ''')
        
        # User information and preferences
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_info (
                key TEXT PRIMARY KEY,
                value TEXT,
                category TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                confidence REAL DEFAULT 1.0
            )
        ''')
        
        # Long-term memories
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                importance REAL DEFAULT 0.5,
                category TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Reminders and tasks
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                due_date DATETIME,
                completed BOOLEAN DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_conversation(self, user_input: str, response: str, emotion: str = "neutral", topics: List[str] = None):
        """Save a conversation with metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        topics_str = json.dumps(topics) if topics else "[]"
        cursor.execute(
            'INSERT INTO conversations (user_input, assistant_response, emotion, topics) VALUES (?, ?, ?, ?)',
            (user_input, response, emotion, topics_str)
        )
        
        self.short_term_memory.append({
            "user": user_input,
            "assistant": response,
            "timestamp": datetime.datetime.now()
        })
        
        conn.commit()
        conn.close()
    
    def get_recent_conversations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve recent conversations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_input, assistant_response, timestamp, emotion, topics 
            FROM conversations 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{
            "user": row[0],
            "assistant": row[1], 
            "timestamp": row[2],
            "emotion": row[3],
            "topics": json.loads(row[4]) if row[4] else []
        } for row in reversed(results)]
    
    def save_user_info(self, key: str, value: str, category: str = "general", confidence: float = 1.0):
        """Save information about the user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT OR REPLACE INTO user_info (key, value, category, confidence) VALUES (?, ?, ?, ?)',
            (key, value, category, confidence)
        )
        
        conn.commit()
        conn.close()
    
    def get_user_info(self, key: str = None, category: str = None) -> Dict[str, Any]:
        """Retrieve user information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if key:
            cursor.execute('SELECT value, confidence FROM user_info WHERE key = ?', (key,))
            result = cursor.fetchone()
            conn.close()
            return {"value": result[0], "confidence": result[1]} if result else None
        
        if category:
            cursor.execute('SELECT key, value, confidence FROM user_info WHERE category = ?', (category,))
        else:
            cursor.execute('SELECT key, value, confidence, category FROM user_info')
        
        results = cursor.fetchall()
        conn.close()
        
        return {row[0]: {"value": row[1], "confidence": row[2]} for row in results}
    
    def add_memory(self, content: str, importance: float = 0.5, category: str = "general"):
        """Add a long-term memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO memories (content, importance, category) VALUES (?, ?, ?)',
            (content, importance, category)
        )
        
        conn.commit()
        conn.close()
    
    def search_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search memories by content"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT content, importance, category, timestamp 
            FROM memories 
            WHERE content LIKE ? 
            ORDER BY importance DESC, timestamp DESC 
            LIMIT ?
        ''', (f'%{query}%', limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{
            "content": row[0],
            "importance": row[1],
            "category": row[2],
            "timestamp": row[3]
        } for row in results]

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

from setuptools import setup, find_packages

setup(
    name="enhanced-voice-assistant",
    version="2.0.0",
    description="Enhanced voice assistant with personality, memory, and discrete mode",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "vosk>=0.3.45",
        "pvrecorder>=1.2.2",
        "ollama>=0.1.7",
    ],
    entry_points={
        "console_scripts": [
            "voice-assistant=main:main",
        ],
    },
    python_requires=">=3.8",
)

#!/usr/bin/env python3
"""
Train a personalized wake word model for the voice assistant
"""

import argparse
import os
from voice_training import WakeWordTrainer

def main():
    parser = argparse.ArgumentParser(
        description="Train a personalized wake word model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train default wake word
  python train_wake_word.py
  
  # Train custom wake word
  python train_wake_word.py --wake-word "jarvis"
  
  # Train with more samples
  python train_wake_word.py --positive-samples 30 --negative-samples 30
        """
    )
    
    parser.add_argument('--wake-word', type=str, default='hades',
                      help='Wake word to train (default: hades)')
    parser.add_argument('--positive-samples', type=int, default=20,
                      help='Number of positive samples to collect (default: 20)')
    parser.add_argument('--negative-samples', type=int, default=20,
                      help='Number of negative samples to collect (default: 20)')
    parser.add_argument('--test-only', action='store_true',
                      help='Test existing model without training')
    
    args = parser.parse_args()
    
    trainer = WakeWordTrainer(args.wake_word)
    
    try:
        if args.test_only:
            # Load and test existing model
            model_path = f"models/wake_word_{args.wake_word}.pkl"
            if os.path.exists(model_path):
                import pickle
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                print(f"✅ Loaded model for '{args.wake_word}'")
                trainer.test_model(model_data)
            else:
                print(f"❌ No trained model found for '{args.wake_word}'")
                print(f"   Run without --test-only to train a new model")
        else:
            # Collect samples
            positive_samples, negative_samples = trainer.collect_samples(
                num_positive=args.positive_samples,
                num_negative=args.negative_samples
            )
            
            # Train model
            if positive_samples and negative_samples:
                model_data = trainer.train_model(positive_samples, negative_samples)
                
                # Test model
                print("\n" + "="*50)
                print("Would you like to test the model now? (recommended)")
                if input("Test model? (y/n): ").lower() == 'y':
                    trainer.test_model(model_data)
            else:
                print("❌ Training cancelled - insufficient samples collected")
                
    except KeyboardInterrupt:
        print("\n\n⚡ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        trainer.cleanup()
        print("\n✅ Cleanup complete")


if __name__ == "__main__":
    main()

import re
import datetime
from typing import Dict, Any, List

def clean_response(text: str) -> str:
    """Remove think tags and clean response"""
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()

def extract_information(user_input: str) -> Dict[str, Any]:
    """Extract user information and topics from input"""
    lower_input = user_input.lower()
    extracted = {
        "topics": [],
        "user_info": {},
        "intent": None
    }
    
    # Extract name
    name_patterns = [
        r"my name is (\w+)",
        r"i'm (\w+)",
        r"i am (\w+)",
        r"call me (\w+)"
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, lower_input)
        if match:
            name = match.group(1).capitalize()
            extracted["user_info"]["name"] = name
            break
    
    # Extract preferences
    if "i like" in lower_input:
        like_match = re.search(r"i like (.+?)(?:\.|,|$)", lower_input)
        if like_match:
            preference = like_match.group(1).strip()
            extracted["topics"].append("preferences")
            extracted["user_info"]["preference"] = preference
    
    # Extract age
    age_match = re.search(r"i am (\d+) years old|i'm (\d+)|my age is (\d+)", lower_input)
    if age_match:
        age = next(g for g in age_match.groups() if g)
        extracted["user_info"]["age"] = age
    
    # Detect topics
    topic_keywords = {
        "weather": ["weather", "temperature", "rain", "sunny", "cloudy"],
        "technology": ["computer", "program", "code", "tech", "software"],
        "music": ["music", "song", "listen", "band", "artist"],
        "food": ["eat", "food", "hungry", "cook", "recipe"],
        "health": ["health", "exercise", "sleep", "tired", "sick"]
    }
    
    for topic, keywords in topic_keywords.items():
        if any(keyword in lower_input for keyword in keywords):
            extracted["topics"].append(topic)
    
    return extracted

def get_time_greeting() -> str:
    """Get appropriate greeting based on time of day"""
    hour = datetime.datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 17:
        return "Good afternoon"
    else:
        return "Good evening"

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string"""
    minutes = int(seconds / 60)
    if minutes == 0:
        return "less than a minute"
    elif minutes == 1:
        return "about a minute"
    else:
        return f"about {minutes} minutes"
import os
import json
import numpy as np
import struct
import pickle
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import librosa
import soundfile as sf
from pvrecorder import PvRecorder
from datetime import datetime

class WakeWordTrainer:
    """Train a personalized wake word detector using voice samples"""
    
    def __init__(self, wake_word: str, sample_dir: str = "voice_samples"):
        self.wake_word = wake_word
        self.sample_dir = sample_dir
        self.model_path = f"models/wake_word_{wake_word}.pkl"
        self.recorder = PvRecorder(device_index=-1, frame_length=512)
        
        # Create directories
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
    def collect_samples(self, num_positive: int = 20, num_negative: int = 20):
        """Collect voice samples for training"""
        print(f"""
        ╔══════════════════════════════════════════╗
        ║        Wake Word Training System         ║
        ╚══════════════════════════════════════════╝
        
        We'll collect voice samples to personalize wake word detection.
        """)
        
        # Collect positive samples (saying the wake word)
        print(f"\n📍 Part 1: Say '{self.wake_word}' {num_positive} times")
        print("Press ENTER to start each recording, speak clearly and naturally.\n")
        
        positive_samples = []
        for i in range(num_positive):
            input(f"Press ENTER to record sample {i+1}/{num_positive}: ")
            audio_data = self._record_sample(duration=2)
            if audio_data is not None and audio_data.size > 0:
                positive_samples.append(audio_data)
                filename = os.path.join(self.sample_dir, f"{self.wake_word}_positive_{i}.wav")
                self._save_audio(audio_data, filename)
                print(f"✅ Saved positive sample {i+1}")
        
        # Collect negative samples (random words/phrases)
        print(f"\n📍 Part 2: Say DIFFERENT words (NOT '{self.wake_word}') {num_negative} times")
        print("Say random words, names, or short phrases.\n")
        
        negative_samples = []
        example_words = ["hello", "computer", "weather", "music", "okay", "yes", "no", "stop"]
        
        for i in range(num_negative):
            suggestion = example_words[i % len(example_words)] if i < len(example_words) else "any word"
            input(f"Press ENTER to record sample {i+1}/{num_negative} (try: '{suggestion}'): ")
            audio_data = self._record_sample(duration=2)
            if audio_data is not None and audio_data.size > 0:
                negative_samples.append(audio_data)
                filename = os.path.join(self.sample_dir, f"negative_{i}.wav")
                self._save_audio(audio_data, filename)
                print(f"✅ Saved negative sample {i+1}")
        
        print("\n✅ Sample collection complete!")
        return positive_samples, negative_samples
    
    def _record_sample(self, duration: float = 2.0) -> np.ndarray:
        """Record a single audio sample"""
        try:
            self.recorder.start()
            print(f"🎤 Recording for {duration} seconds... Speak now!")
            
            frames = []
            num_frames = int(self.recorder.sample_rate * duration / self.recorder.frame_length)
            
            for _ in range(num_frames):
                frames.extend(self.recorder.read())
            
            self.recorder.stop()
            print("⏹️  Recording complete")
            
            # Convert to numpy array
            audio_data = np.array(frames, dtype=np.float32) / 32768.0  # Normalize
            return audio_data
            
        except Exception as e:
            print(f"Error recording: {e}")
            return None
    
    def _save_audio(self, audio_data: np.ndarray, filename: str):
        """Save audio data to file"""
        sf.write(filename, audio_data, self.recorder.sample_rate)
    
    def extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio"""
        # Use librosa for feature extraction
        mfccs = librosa.feature.mfcc(
            y=audio_data, 
            sr=self.recorder.sample_rate,
            n_mfcc=13,
            hop_length=512
        )
        
        # Add delta and delta-delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Concatenate all features
        features = np.concatenate([
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.mean(delta_mfccs, axis=1),
            np.mean(delta2_mfccs, axis=1)
        ])
        
        return features
    
    def train_model(self, positive_samples: List[np.ndarray], negative_samples: List[np.ndarray]):
        """Train the wake word detection model"""
        print("\n🧠 Training personalized wake word model...")
        
        # Extract features
        X = []
        y = []
        
        print("Extracting features from positive samples...")
        for sample in positive_samples:
            features = self.extract_features(sample)
            X.append(features)
            y.append(1)  # Positive class
        
        print("Extracting features from negative samples...")
        for sample in negative_samples:
            features = self.extract_features(sample)
            X.append(features)
            y.append(0)  # Negative class
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train SVM classifier
        print("Training classifier...")
        model = SVC(kernel='rbf', probability=True, gamma='auto')
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        print(f"\n📊 Training Results:")
        print(f"  Training accuracy: {train_score:.2%}")
        print(f"  Testing accuracy: {test_score:.2%}")
        
        # Save model
        model_data = {
            'model': model,
            'scaler': scaler,
            'wake_word': self.wake_word,
            'sample_rate': self.recorder.sample_rate,
            'trained_at': datetime.now().isoformat()
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to {self.model_path}")
        return model_data
    
    def test_model(self, model_data: Dict[str, Any]):
        """Test the trained model interactively"""
        print("\n🧪 Testing wake word detection...")
        print("Say the wake word or other words to test. Press Ctrl+C to stop.\n")
        
        model = model_data['model']
        scaler = model_data['scaler']
        
        try:
            while True:
                input("Press ENTER to test: ")
                audio_data = self._record_sample(duration=2)
                
                if audio_data is not None and audio_data.size > 0:
                    # Extract features
                    features = self.extract_features(audio_data).reshape(1, -1)
                    features_scaled = scaler.transform(features)
                    
                    # Predict
                    prediction = model.predict(features_scaled)[0]
                    probability = model.predict_proba(features_scaled)[0]
                    
                    if prediction == 1:
                        print(f"✅ Wake word detected! (confidence: {probability[1]:.2%})")
                    else:
                        print(f"❌ Not wake word (confidence: {probability[0]:.2%})")
                
        except KeyboardInterrupt:
            print("\n\nTesting stopped.")
    
    def cleanup(self):
        """Clean up resources"""
        if self.recorder:
            self.recorder.delete()


class PersonalizedWakeWordDetector:
    """Use trained model for wake word detection"""
    
    def __init__(self, model_path: str):
        with open(model_path, 'rb') as f:
            self.model_data = pickle.load(f)
        
        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.wake_word = self.model_data['wake_word']
        self.sample_rate = self.model_data['sample_rate']
        
        # Confidence threshold
        self.threshold = 0.7
    
    def is_wake_word(self, audio_data: np.ndarray) -> tuple[bool, float]:
        """Check if audio contains wake word"""
        try:
            # Extract features
            features = self.extract_features(audio_data).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            # Predict
            probability = self.model.predict_proba(features_scaled)[0]
            confidence = probability[1]  # Probability of positive class
            
            return confidence >= self.threshold, confidence
            
        except Exception as e:
            print(f"Error in wake word detection: {e}")
            return False, 0.0
    
    def extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract the same features used in training"""
        mfccs = librosa.feature.mfcc(
            y=audio_data, 
            sr=self.sample_rate,
            n_mfcc=13,
            hop_length=512
        )
        
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        features = np.concatenate([
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.mean(delta_mfccs, axis=1),
            np.mean(delta2_mfccs, axis=1)
        ])
        
        return features


# Training script
def train_wake_word(wake_word: str = "hades"):
    """Interactive training script"""
    trainer = WakeWordTrainer(wake_word)
    
    try:
        # Collect samples
        positive_samples, negative_samples = trainer.collect_samples(
            num_positive=20,
            num_negative=20
        )
        
        # Train model
        model_data = trainer.train_model(positive_samples, negative_samples)
        
        # Test model
        trainer.test_model(model_data)
        
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    import sys
    
    wake_word = sys.argv[1] if len(sys.argv) > 1 else "hades"
    train_wake_word(wake_word)

"""
Advanced wake word optimization with multiple detection strategies
"""

import numpy as np
from typing import List, Tuple, Optional
import json
import os
from dataclasses import dataclass
from collections import deque

@dataclass
class WakeWordConfig:
    """Configuration for wake word detection"""
    # Detection methods to use
    use_personalized: bool = True
    use_vosk: bool = True
    use_keyword_spotting: bool = False
    
    # Thresholds
    personalized_threshold: float = 0.7
    vosk_confidence_boost: float = 0.2  # Boost confidence if both methods agree
    
    # Sliding window for continuous detection
    window_size_ms: int = 2000
    hop_size_ms: int = 500
    
    # Adaptive thresholds
    enable_adaptive_threshold: bool = True
    false_positive_penalty: float = 0.05
    false_negative_bonus: float = 0.02


class OptimizedWakeWordDetector:
    """Multi-strategy wake word detector with optimization"""
    
    def __init__(self, config: WakeWordConfig, wake_word: str):
        self.config = config
        self.wake_word = wake_word
        self.detection_history = deque(maxlen=50)
        self.current_threshold = config.personalized_threshold
        
        # Load models
        self.personalized_detector = None
        self.vosk_recognizer = None
        
        self._load_models()
    
    def _load_models(self):
        """Load all available detection models"""
        # Load personalized model
        if self.config.use_personalized:
            model_path = f"models/wake_word_{self.wake_word}.pkl"
            if os.path.exists(model_path):
                try:
                    from voice_training import PersonalizedWakeWordDetector
                    self.personalized_detector = PersonalizedWakeWordDetector(model_path)
                    # Override threshold
                    self.personalized_detector.threshold = self.current_threshold
                    print(f"✅ Loaded personalized model for '{self.wake_word}'")
                except Exception as e:
                    print(f"⚠️  Could not load personalized model: {e}")
    
    def detect(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Tuple[bool, float, dict]:
        """
        Detect wake word using multiple strategies
        
        Returns:
            - is_detected: bool
            - confidence: float (0.0 to 1.0)
            - details: dict with detection details
        """
        results = {
            'personalized': {'detected': False, 'confidence': 0.0},
            'vosk': {'detected': False, 'confidence': 0.0},
            'combined': {'detected': False, 'confidence': 0.0}
        }
        
        # Personalized detection
        if self.personalized_detector and self.config.use_personalized:
            detected, confidence = self.personalized_detector.is_wake_word(audio_data)
            results['personalized'] = {
                'detected': detected,
                'confidence': confidence
            }
        
        # Combine results
        combined_confidence = self._combine_confidences(results)
        is_detected = combined_confidence >= self.current_threshold
        
        results['combined'] = {
            'detected': is_detected,
            'confidence': combined_confidence,
            'threshold': self.current_threshold
        }
        
        # Update adaptive threshold
        if self.config.enable_adaptive_threshold:
            self._update_threshold(is_detected, combined_confidence)
        
        # Record detection
        self.detection_history.append({
            'detected': is_detected,
            'confidence': combined_confidence,
            'timestamp': np.datetime64('now')
        })
        
        return is_detected, combined_confidence, results
    
    def _combine_confidences(self, results: dict) -> float:
        """Combine confidences from multiple detectors"""
        confidences = []
        
        if results['personalized']['confidence'] > 0:
            confidences.append(results['personalized']['confidence'])
        
        if not confidences:
            return 0.0
        
        # Weighted average with boost if multiple agree
        base_confidence = np.mean(confidences)
        
        # Boost if multiple methods agree
        agreement_count = sum(1 for r in results.values() 
                            if isinstance(r, dict) and r.get('detected', False))
        
        if agreement_count >= 2:
            base_confidence += self.config.vosk_confidence_boost
        
        return min(base_confidence, 1.0)
    
    def _update_threshold(self, detected: bool, confidence: float):
        """Adaptively update detection threshold based on user feedback"""
        # This is a simplified adaptive algorithm
        # In practice, you'd want user feedback on false positives/negatives
        
        if detected and confidence > 0.9:
            # Very confident detection - slightly lower threshold
            self.current_threshold *= (1 - self.config.false_negative_bonus)
        elif not detected and confidence > self.current_threshold * 0.8:
            # Close miss - might be false negative
            self.current_threshold *= (1 - self.config.false_negative_bonus * 0.5)
        
        # Keep threshold in reasonable range
        self.current_threshold = np.clip(self.current_threshold, 0.5, 0.9)
        
        if self.personalized_detector:
            self.personalized_detector.threshold = self.current_threshold
    
    def get_statistics(self) -> dict:
        """Get detection statistics"""
        if not self.detection_history:
            return {}
        
        detections = [d['detected'] for d in self.detection_history]
        confidences = [d['confidence'] for d in self.detection_history]
        
        return {
            'total_attempts': len(self.detection_history),
            'detections': sum(detections),
            'detection_rate': sum(detections) / len(detections) if detections else 0,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'current_threshold': self.current_threshold
        }


class ContinuousWakeWordDetector:
    """Continuous wake word detection with sliding window"""
    
    def __init__(self, detector: OptimizedWakeWordDetector, sample_rate: int = 16000):
        self.detector = detector
        self.sample_rate = sample_rate
        self.buffer_size = int(detector.config.window_size_ms * sample_rate / 1000)
        self.hop_size = int(detector.config.hop_size_ms * sample_rate / 1000)
        self.audio_buffer = deque(maxlen=self.buffer_size)
        
    def process_audio_stream(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """Process audio chunk and detect wake word"""
        # Add to buffer
        self.audio_buffer.extend(audio_chunk)
        
        # Check if we have enough data
        if len(self.audio_buffer) < self.buffer_size:
            return False, 0.0
        
        # Convert to numpy array
        audio_window = np.array(list(self.audio_buffer))
        
        # Detect wake word
        detected, confidence, details = self.detector.detect(audio_window, self.sample_rate)
        
        return detected, confidence