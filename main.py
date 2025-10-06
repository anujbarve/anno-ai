# main.py
#!/usr/bin/env python3
"""
Enhanced Voice Assistant - Main Entry Point
Supports both voice and text interfaces
"""

import os
import sys
import argparse
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