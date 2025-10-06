import os
import struct
import datetime
import random
import json
import sqlite3
import threading
import time
import pickle
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from collections import deque
from vosk import Model, KaldiRecognizer
from pvrecorder import PvRecorder
import subprocess
import ollama
import re

# --- Configuration ---
@dataclass
class AssistantConfig:
    """Configuration for the voice assistant"""
    wake_word: str = "hades"
    command_record_seconds: int = 7
    wake_word_record_seconds: int = 2
    vosk_model_path: str = "models/vosk-model-en-in-0.5"
    piper_model_path: str = "models/en_US-lessac-medium.onnx"
    piper_sample_rate: str = "22050"
    ollama_model: str = "qwen3:0.6b"
    memory_db_path: str = "assistant_memory.db"
    max_conversation_history: int = 20
    
    # Personality Configuration
    assistant_name: str = "Hades"
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

class ConversationMemory:
    """Manages persistent memory and conversation history"""
    
    def __init__(self, db_path: str, max_history: int = 20):
        self.db_path = db_path
        self.max_history = max_history
        self.short_term_memory = deque(maxlen=10)  # Recent context
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
        
        # Add to short-term memory
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

class FeatureManager:
    """Manages additional features and capabilities"""
    
    def __init__(self, memory: ConversationMemory):
        self.memory = memory
        self.active_timers = {}
        
    def set_reminder(self, content: str, time_str: str) -> str:
        """Set a reminder (simplified version)"""
        # This is a placeholder - in a real implementation, 
        # you'd parse the time and set actual reminders
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
            "Mythology fact: In Greek mythology, Hades' three-headed dog Cerberus guards the underworld."
        ]
        return random.choice(facts)

class EnhancedVoiceAssistant:
    """Main voice assistant class with enhanced features"""
    
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.memory = ConversationMemory(config.memory_db_path, config.max_conversation_history)
        self.personality = PersonalityEngine(config)
        self.features = FeatureManager(self.memory)
        self.session_start = datetime.datetime.now()
        self.interaction_count = 0
        
        # Initialize speech recognition
        if not os.path.exists(config.vosk_model_path):
            raise FileNotFoundError(f"Vosk model not found at '{config.vosk_model_path}'")
        
        self.model = Model(config.vosk_model_path)
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.recorder = PvRecorder(device_index=-1, frame_length=512)
        
    def speak(self, text: str, emotion: str = "neutral"):
        """Enhanced speak function with emotion support"""
        if not text.strip():
            return
            
        # Add emotional cues to text
        emotional_prefix = {
            "happy": "😊 ",
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
    
    def clean_response(self, text: str) -> str:
        """Remove think tags and clean response"""
        cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return cleaned_text.strip()
    
    def extract_information(self, user_input: str) -> Dict[str, Any]:
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
                self.memory.save_user_info("name", name, "personal", 1.0)
                break
        
        # Extract preferences
        if "i like" in lower_input:
            like_match = re.search(r"i like (.+?)(?:\.|,|$)", lower_input)
            if like_match:
                preference = like_match.group(1).strip()
                extracted["topics"].append("preferences")
                self.memory.save_user_info(f"likes_{len(extracted['topics'])}", preference, "preferences", 0.9)
        
        # Extract age
        age_match = re.search(r"i am (\d+) years old|i'm (\d+)|my age is (\d+)", lower_input)
        if age_match:
            age = next(g for g in age_match.groups() if g)
            extracted["user_info"]["age"] = age
            self.memory.save_user_info("age", age, "personal", 0.95)
        
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
    
    def generate_contextual_prompt(self, user_input: str) -> str:
        """Generate a context-aware prompt for the LLM"""
        # Get conversation history
        recent_convos = self.memory.get_recent_conversations(3)
        
        # Get user information
        user_info = self.memory.get_user_info()
        user_name = user_info.get("name", {}).get("value", "friend")
        
        # Build context
        context_parts = [self.personality.get_personality_prompt()]
        
        # Add user info context
        if user_info:
            info_summary = []
            for key, data in user_info.items():
                if data["confidence"] > 0.7:
                    info_summary.append(f"{key}: {data['value']}")
            if info_summary:
                context_parts.append(f"Known user information: {', '.join(info_summary[:5])}")
        
        # Add conversation history
        if recent_convos:
            history = "\nRecent conversation:\n"
            for conv in recent_convos[-3:]:
                history += f"User: {conv['user']}\nYou: {conv['assistant']}\n"
            context_parts.append(history)
        
        # Add session info
        session_duration = (datetime.datetime.now() - self.session_start).total_seconds() / 60
        context_parts.append(f"Session duration: {session_duration:.1f} minutes")
        
        # Combine everything
        full_prompt = "\n".join(context_parts) + f"\n\nUser: {user_input}\n\nRespond naturally and concisely:"
        
        return full_prompt
    
    def ask_ollama(self, prompt: str) -> Tuple[str, str]:
        """Enhanced Ollama interaction with context and personality"""
        try:
            self.interaction_count += 1
            
            # Extract information from user input
            extracted_info = self.extract_information(prompt)
            
            # Update mood based on input
            self.personality.update_mood(prompt, extracted_info)
            
            # Generate contextual prompt
            full_prompt = self.generate_contextual_prompt(prompt)
            
            print(f"\n💭 Processing request...")
            
            response = ollama.chat(
                model=self.config.ollama_model,
                messages=[
                    {'role': 'system', 'content': full_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                stream=False
            )
            
            raw_response = response['message']['content']
            cleaned_response = self.clean_response(raw_response)
            
            # Save conversation with metadata
            self.memory.add_conversation(
                prompt, 
                cleaned_response,
                self.personality.current_mood,
                extracted_info["topics"]
            )
            
            return cleaned_response, self.personality.current_mood
            
        except Exception as e:
            print(f"Error communicating with Ollama: {e}")
            return "I'm having trouble processing that. Could you try again?", "neutral"
    
    def process_command(self, command: str) -> bool:
        """Enhanced command processing with advanced features"""
        if not command.strip():
            self.speak("I didn't catch that. Could you try again?")
            return False
        
        command_lower = command.lower().strip()
        
        # System commands
        if any(word in command_lower for word in ["goodbye", "exit", "quit", "stop", "shutdown"]):
            farewell_messages = [
                f"Goodbye! We talked {self.interaction_count} times today. See you soon!",
                "Take care! I'll remember our conversation.",
                "Farewell! It was great chatting with you.",
                "Until next time! I enjoyed our chat."
            ]
            self.speak(random.choice(farewell_messages), "neutral")
            return True
        
        # Memory-related commands
        elif "what do you know about me" in command_lower:
            user_info = self.memory.get_user_info()
            if user_info:
                info_parts = []
                if "name" in user_info:
                    info_parts.append(f"Your name is {user_info['name']['value']}")
                if "age" in user_info:
                    info_parts.append(f"you're {user_info['age']['value']} years old")
                
                # Get preferences
                prefs = [v['value'] for k, v in user_info.items() if k.startswith('likes_')]
                if prefs:
                    info_parts.append(f"you like {', '.join(prefs[:3])}")
                
                if info_parts:
                    response = f"I know that {', and '.join(info_parts)}."
                else:
                    response = "I'm still getting to know you! Tell me more about yourself."
            else:
                response = "We haven't talked much yet. Tell me about yourself!"
            
            self.speak(response, "friendly")
            return False
        
        # Fun features
        elif "tell me a joke" in command_lower:
            jokes = [
                "Why don't scientists trust atoms? Because they make up everything!",
                "What did the grape say when it got stepped on? Nothing, it just let out a little wine!",
                "Why did the scarecrow win an award? He was outstanding in his field!",
                "I told my wife she was drawing her eyebrows too high. She looked surprised!",
                "Why don't eggs tell jokes? They'd crack up!",
                "What's the best thing about Switzerland? I don't know, but the flag is a big plus!"
            ]
            self.speak(random.choice(jokes), "playful")
            return False
        
        elif "fun fact" in command_lower or "interesting fact" in command_lower:
            self.speak(self.features.get_fun_fact(), "cheerful")
            return False
        
        # Time and date
        elif "time" in command_lower and "what" in command_lower:
            current_time = datetime.datetime.now()
            time_str = current_time.strftime("%I:%M %p")
            responses = [
                f"It's {time_str} right now.",
                f"The time is {time_str}.",
                f"Currently, it's {time_str}."
            ]
            self.speak(random.choice(responses))
            return False
        
        elif "date" in command_lower and ("what" in command_lower or "today" in command_lower):
            today = datetime.datetime.now()
            date_str = today.strftime("%B %d, %Y")
            day_name = today.strftime("%A")
            self.speak(f"Today is {day_name}, {date_str}.")
            return False
        
        # Reminder feature
        elif "remind me" in command_lower:
            # Extract reminder content (simplified)
            reminder_match = re.search(r"remind me to (.+?)(?:at|in|$)", command_lower)
            if reminder_match:
                reminder_content = reminder_match.group(1).strip()
                response = self.features.set_reminder(reminder_content, "later")
                self.speak(response)
            else:
                self.speak("What would you like me to remind you about?")
            return False
        
        # Session information
        elif "how long" in command_lower and "talking" in command_lower:
            duration = datetime.datetime.now() - self.session_start
            minutes = int(duration.total_seconds() / 60)
            if minutes == 0:
                time_str = "less than a minute"
            elif minutes == 1:
                time_str = "about a minute"
            else:
                time_str = f"about {minutes} minutes"
            
            self.speak(f"We've been chatting for {time_str}, with {self.interaction_count} exchanges.")
            return False
        
        # Default: Use Ollama for general conversation
        else:
            response, mood = self.ask_ollama(command)
            if response:
                self.speak(response, mood)
            return False
    
    def transcribe_audio(self, duration: int) -> Optional[str]:
        """Enhanced audio transcription"""
        try:
            self.recorder.start()
            print(f"🎤 Listening for {duration} seconds...")
            
            # Visual feedback
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
    
    def run(self):
        """Main run loop with enhanced features"""
        try:
            # Startup greeting based on user history
            user_info = self.memory.get_user_info("name")
            user_name = user_info["value"] if user_info else None
            
            # Check time of day for appropriate greeting
            hour = datetime.datetime.now().hour
            if hour < 12:
                time_greeting = "Good morning"
            elif hour < 17:
                time_greeting = "Good afternoon"
            else:
                time_greeting = "Good evening"
            
            if user_name:
                # Check last conversation
                last_convos = self.memory.get_recent_conversations(1)
                if last_convos:
                    last_time = datetime.datetime.fromisoformat(last_convos[0]['timestamp'])
                    days_ago = (datetime.datetime.now() - last_time).days
                    
                    if days_ago == 0:
                        greeting = f"Welcome back, {user_name}! Ready for more?"
                    elif days_ago == 1:
                        greeting = f"{time_greeting}, {user_name}! It's good to hear from you again."
                    else:
                        greeting = f"{time_greeting}, {user_name}! It's been {days_ago} days. I've missed our chats!"
                else:
                    greeting = f"{time_greeting}, {user_name}! I'm {self.config.assistant_name}, ready to help."
            else:
                greeting = f"{time_greeting}! I'm {self.config.assistant_name}, your voice assistant. Say my name when you need me."
            
            self.speak(greeting, "cheerful")
            
            consecutive_fails = 0
            
            while True:
                try:
                    # Show different waiting messages
                    waiting_messages = [
                        f"👂 Listening for '{self.config.wake_word}'...",
                        f"🎯 Ready when you say '{self.config.wake_word}'...",
                        f"💫 Waiting for the magic word '{self.config.wake_word}'...",
                        f"🔊 Say '{self.config.wake_word}' to wake me..."
                    ]
                    print(f"\n{random.choice(waiting_messages)}")
                    
                    # Listen for wake word
                    wake_text = self.transcribe_audio(self.config.wake_word_record_seconds)
                    
                    if wake_text and self.config.wake_word in wake_text:
                        consecutive_fails = 0
                        print(f"✨ Wake word detected!")
                        
                        # Varied acknowledgments based on mood
                        acknowledgments = {
                            "cheerful": ["Yes! What can I do for you?", "Here and happy to help!", "At your service!"],
                            "neutral": ["Yes? How can I help?", "I'm listening.", "What do you need?"],
                            "thoughtful": ["Yes, I'm here. What's on your mind?", "How can I assist you?", "I'm listening carefully."],
                            "playful": ["You rang? 😊", "Present and accounted for!", "Your wish is my command!"]
                        }
                        
                        response = random.choice(acknowledgments.get(self.personality.current_mood, acknowledgments["neutral"]))
                        self.speak(response, self.personality.current_mood)
                        
                        # Listen for command
                        command = self.transcribe_audio(self.config.command_record_seconds)
                        
                        if command:
                            print(f"💬 Processing: \"{command}\"")
                            if self.process_command(command):
                                break
                        else:
                            no_input_responses = [
                                "I didn't catch that. Could you try again?",
                                "Sorry, I missed that. What did you say?",
                                "Could you repeat that, please?",
                                "I'm having trouble hearing you. One more time?"
                            ]
                            self.speak(random.choice(no_input_responses))
                            consecutive_fails += 1
                    
                    # Handle multiple failures
                    if consecutive_fails >= 3:
                        self.speak("I'm having trouble hearing you. Make sure your microphone is working properly.")
                        consecutive_fails = 0
                        
                except KeyboardInterrupt:
                    print("\n⚡ Interrupt detected!")
                    break
                except Exception as e:
                    print(f"❌ Error in main loop: {e}")
                    self.speak("I encountered an error. Let me reset and try again.")
                    time.sleep(2)
                    
        except Exception as e:
            print(f"❌ Fatal error: {e}")
            self.speak("I'm experiencing a critical error and need to shut down.")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\n🔧 Cleaning up...")
        if hasattr(self, 'recorder') and self.recorder:
            self.recorder.delete()
        
        # Save session summary
        session_duration = (datetime.datetime.now() - self.session_start).total_seconds() / 60
        summary = f"Session lasted {session_duration:.1f} minutes with {self.interaction_count} interactions"
        self.memory.add_memory(summary, importance=0.3, category="session")
        
        print("👋 Goodbye!")

# Additional utility functions

def create_assistant_with_custom_personality(name: str = "Hades", 
                                            traits: List[str] = None,
                                            voice_model: str = None) -> EnhancedVoiceAssistant:
    """Factory function to create assistant with custom personality"""
    config = AssistantConfig()
    
    if name:
        config.assistant_name = name
        config.wake_word = name.lower()
    
    if traits:
        config.personality_traits["core_traits"] = traits
    
    if voice_model and os.path.exists(voice_model):
        config.piper_model_path = voice_model
    
    return EnhancedVoiceAssistant(config)

# Main entry point
def main():
    """Enhanced main function with better error handling and configuration"""
    print("""
    ╔══════════════════════════════════════════╗
    ║      Enhanced Voice Assistant v2.0       ║
    ║         Powered by Local AI              ║
    ╚══════════════════════════════════════════╝
    """)
    
    # Check for required files
    config = AssistantConfig()
    
    required_files = {
        "Vosk Model": config.vosk_model_path,
        "Piper Model": config.piper_model_path
    }
    
    print("🔍 Checking required files...")
    missing_files = []
    
    for name, path in required_files.items():
        if os.path.exists(path):
            print(f"✅ {name}: Found")
        else:
            print(f"❌ {name}: Missing at {path}")
            missing_files.append(name)
    
    if missing_files:
        print(f"\n⚠️  Missing required files: {', '.join(missing_files)}")
        print("Please download the required models and place them in the correct directories.")
        return
    
    # Check Ollama connection
    print("\n🔌 Checking Ollama connection...")
    try:
        # Test Ollama connection
        test_response = ollama.chat(
            model=config.ollama_model,
            messages=[{'role': 'user', 'content': 'test'}],
            stream=False
        )
        print("✅ Ollama: Connected")
    except Exception as e:
        print(f"❌ Ollama: Failed to connect - {e}")
        print("Please make sure Ollama is running and the model is installed.")
        return
    
    # Initialize and run assistant
    try:
        print("\n🚀 Initializing assistant...")
        assistant = EnhancedVoiceAssistant(config)
        
        print("✨ Assistant ready!\n")
        print(f"Wake word: '{config.wake_word}'")
        print("Say 'goodbye' or press Ctrl+C to exit\n")
        
        assistant.run()
        
    except KeyboardInterrupt:
        print("\n\n⚡ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

# Standalone testing functions

def test_memory_system():
    """Test the memory system independently"""
    print("🧪 Testing memory system...")
    memory = ConversationMemory("test_memory.db")
    
    # Test saving and retrieving user info
    memory.save_user_info("name", "John", "personal")
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

if __name__ == '__main__':
    # Command line argument handling
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test-memory":
            test_memory_system()
        elif sys.argv[1] == "--test-personality":
            test_personality_engine()
        elif sys.argv[1] == "--help":
            print("""
Enhanced Voice Assistant

Usage:
    python assistant.py                 # Run the assistant
    python assistant.py --test-memory   # Test memory system
    python assistant.py --test-personality  # Test personality engine
    python assistant.py --help          # Show this help message

Configuration:
    Edit the AssistantConfig class to customize:
    - Wake word
    - Model paths
    - Personality traits
    - Recording durations
            """)
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for available options")
    else:
        main()