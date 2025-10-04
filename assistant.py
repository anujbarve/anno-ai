import datetime
import random
import ollama
from typing import Optional, Tuple, Dict, Any
from config import AssistantConfig
from memory import ConversationMemory
from personality import PersonalityEngine
from features import FeatureManager
from interfaces import AssistantInterface, VoiceInterface, TextInterface
from utils import clean_response, extract_information, get_time_greeting, format_duration

class EnhancedAssistant:
    """Main assistant class with pluggable interfaces"""
    
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.memory = ConversationMemory(config.memory_db_path, config.max_conversation_history)
        self.personality = PersonalityEngine(config)
        self.features = FeatureManager(self.memory)
        self.session_start = datetime.datetime.now()
        self.interaction_count = 0
        
        # Select interface based on mode
        if config.discrete_mode:
            self.interface: AssistantInterface = TextInterface(config)
        else:
            self.interface: AssistantInterface = VoiceInterface(config)
    
    def initialize(self):
        """Initialize the assistant"""
        self.interface.initialize()
        
        # Get startup greeting
        greeting = self._generate_greeting()
        self.interface.output_response(greeting, "cheerful")
    
    def _generate_greeting(self) -> str:
        """Generate personalized greeting"""
        user_info = self.memory.get_user_info("name")
        user_name = user_info["value"] if user_info else None
        time_greeting = get_time_greeting()
        
        if user_name:
            # Check last conversation
            last_convos = self.memory.get_recent_conversations(1)
            if last_convos:
                last_time = datetime.datetime.fromisoformat(last_convos[0]['timestamp'])
                days_ago = (datetime.datetime.now() - last_time).days
                
                if days_ago == 0:
                    return f"Welcome back, {user_name}! Ready for more?"
                elif days_ago == 1:
                    return f"{time_greeting}, {user_name}! It's good to hear from you again."
                else:
                    return f"{time_greeting}, {user_name}! It's been {days_ago} days. I've missed our chats!"
            else:
                return f"{time_greeting}, {user_name}! I'm {self.config.assistant_name}, ready to help."
        else:
            return f"{time_greeting}! I'm {self.config.assistant_name}, your assistant. Say my name when you need me."
    
    def generate_contextual_prompt(self, user_input: str) -> str:
        """Generate a context-aware prompt for the LLM"""
        recent_convos = self.memory.get_recent_conversations(3)
        user_info = self.memory.get_user_info()
        
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
            extracted_info = extract_information(prompt)
            
            # Save any extracted user info
            for key, value in extracted_info["user_info"].items():
                if key == "preference":
                    self.memory.save_user_info(f"likes_{self.interaction_count}", value, "preferences", 0.9)
                else:
                    self.memory.save_user_info(key, value, "personal", 0.95)
            
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
            cleaned_response = clean_response(raw_response)
            
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
        """Process user commands"""
        if not command.strip():
            self.interface.output_response("I didn't catch that. Could you try again?")
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
            self.interface.output_response(random.choice(farewell_messages), "neutral")
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
            
            self.interface.output_response(response, "friendly")
            return False
        
        # Fun features
        elif "tell me a joke" in command_lower:
            jokes = [
                "Why don't scientists trust atoms? Because they make up everything!",
                "What did the grape say when it got stepped on? Nothing, it just let out a little wine!",
                "Why did the scarecrow win an award? He was outstanding in his field!",
                "I told my wife she was drawing her eyebrows too high. She looked surprised!",
                "Why don't eggs tell jokes? They'd crack up!",
                "What's the best thing about Switzerland? I don't know, but the flag is a big plus!",
                "How do you organize a space party? You planet!",
                "Why did the bicycle fall over? It was two-tired!"
            ]
            self.interface.output_response(random.choice(jokes), "playful")
            return False
        
        elif "fun fact" in command_lower or "interesting fact" in command_lower:
            self.interface.output_response(self.features.get_fun_fact(), "cheerful")
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
            self.interface.output_response(random.choice(responses))
            return False
        
        elif "date" in command_lower and ("what" in command_lower or "today" in command_lower):
            today = datetime.datetime.now()
            date_str = today.strftime("%B %d, %Y")
            day_name = today.strftime("%A")
            self.interface.output_response(f"Today is {day_name}, {date_str}.")
            return False
        
        # Reminder feature
        elif "remind me" in command_lower:
            import re
            reminder_match = re.search(r"remind me to (.+?)(?:at|in|$)", command_lower)
            if reminder_match:
                reminder_content = reminder_match.group(1).strip()
                response = self.features.set_reminder(reminder_content, "later")
                self.interface.output_response(response)
            else:
                self.interface.output_response("What would you like me to remind you about?")
            return False
        
        # Session information
        elif "how long" in command_lower and "talking" in command_lower:
            duration = datetime.datetime.now() - self.session_start
            time_str = format_duration(duration.total_seconds())
            self.interface.output_response(
                f"We've been chatting for {time_str}, with {self.interaction_count} exchanges."
            )
            return False
        
        # Default: Use Ollama for general conversation
        else:
            response, mood = self.ask_ollama(command)
            if response:
                self.interface.output_response(response, mood)
            return False
    
    def run(self):
        """Main run loop"""
        try:
            consecutive_fails = 0
            
            while True:
                try:
                    # Different prompts for discrete vs voice mode
                    if self.config.discrete_mode:
                        # In text mode, just wait for input
                        user_input = self.interface.get_user_input("wake_word")
                        
                        if not user_input:
                            continue
                            
                        # Check for exit commands first
                        if user_input.lower() in ["exit", "quit", "goodbye"]:
                            if self.process_command(user_input):
                                break
                            continue
                        
                        # Check for wake word
                        if self.interface.is_wake_word_detected(user_input):
                            # Extract command from the same input if present
                            wake_lower = user_input.lower()
                            wake_word_lower = self.config.wake_word.lower()
                            
                            # Remove wake word from input to get command
                            if wake_lower == wake_word_lower:
                                # Just the wake word, ask for command
                                self.interface.output_response("Yes? How can I help?", self.personality.current_mood)
                                command = self.interface.get_user_input()
                            else:
                                # Command included with wake word
                                command = user_input.replace(self.config.wake_word, "", 1).strip()
                                if not command:
                                    self.interface.output_response("Yes? How can I help?", self.personality.current_mood)
                                    command = self.interface.get_user_input()
                            
                            if command:
                                if self.process_command(command):
                                    break
                    else:
                        # Voice mode - original behavior
                        print(f"\n👂 Listening for '{self.config.wake_word}'...")
                        
                        wake_text = self.interface.get_user_input("wake_word")
                        
                        if wake_text and self.interface.is_wake_word_detected(wake_text):
                            consecutive_fails = 0
                            print(f"✨ Wake word detected!")
                            
                            # Varied acknowledgments
                            acknowledgments = {
                                "cheerful": ["Yes! What can I do for you?", "Here and happy to help!", "At your service!"],
                                                                "neutral": ["Yes? How can I help?", "I'm listening.", "What do you need?"],
                                "thoughtful": ["Yes, I'm here. What's on your mind?", "How can I assist you?", "I'm listening carefully."],
                                "playful": ["You rang? 😊", "Present and accounted for!", "Your wish is my command!"]
                            }
                            
                            response = random.choice(acknowledgments.get(self.personality.current_mood, acknowledgments["neutral"]))
                            self.interface.output_response(response, self.personality.current_mood)
                            
                            # Listen for command
                            command = self.interface.get_user_input()
                            
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
                                self.interface.output_response(random.choice(no_input_responses))
                                consecutive_fails += 1
                    
                    # Handle multiple failures (voice mode only)
                    if not self.config.discrete_mode and consecutive_fails >= 3:
                        self.interface.output_response("I'm having trouble hearing you. Make sure your microphone is working properly.")
                        consecutive_fails = 0
                        
                except KeyboardInterrupt:
                    print("\n⚡ Interrupt detected!")
                    break
                except Exception as e:
                    print(f"❌ Error in main loop: {e}")
                    self.interface.output_response("I encountered an error. Let me reset and try again.")
                    import time
                    time.sleep(2)
                    
        except Exception as e:
            print(f"❌ Fatal error: {e}")
            self.interface.output_response("I'm experiencing a critical error and need to shut down.")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\n🔧 Cleaning up...")
        
        # Clean up interface
        self.interface.cleanup()
        
        # Save session summary
        session_duration = (datetime.datetime.now() - self.session_start).total_seconds() / 60
        summary = f"Session lasted {session_duration:.1f} minutes with {self.interaction_count} interactions"
        self.memory.add_memory(summary, importance=0.3, category="session")
        
        print("👋 Goodbye!")