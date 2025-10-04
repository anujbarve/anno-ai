# Enhanced Voice Assistant v2.0

A powerful voice assistant with personality, memory, and both voice and text interfaces.

## Features

- **Dual Interface**: Voice mode with STT/TTS or discrete text-only mode
- **Personality System**: Dynamic mood and personality traits
- **Persistent Memory**: Remembers user information and conversations
- **Local AI**: Powered by Ollama for privacy
- **Extensible**: Easy to add new features and commands

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download required models:
   - Vosk model for speech recognition (voice mode only)
   - Piper model for text-to-speech (voice mode only)
   - Install Ollama and pull your preferred model

## Usage

### Voice Mode (Default)
```bash
python main.py
```

### Discrete/Text Mode
```bash
python main.py --discrete
```

### Custom Configuration
```bash
# Custom name and wake word
python main.py --name "Jarvis" --wake-word "jarvis"

# Use different Ollama model
python main.py --model "llama2:13b"

# Custom memory database
python main.py --memory-db "my_assistant.db"
```

### Testing
```bash
# Test memory system
python main.py --test-memory

# Test personality engine
python main.py --test-personality
```

## File Structure

```
assistant/
├── config.py              # Configuration dataclass
├── memory.py              # Memory and conversation management
├── personality.py         # Personality engine
├── features.py            # Additional features (reminders, facts, etc.)
├── interfaces/            # Interface implementations
│   ├── __init__.py
│   ├── base.py           # Abstract interface
│   ├── voice_interface.py # Voice STT/TTS interface
│   └── text_interface.py  # Text-only interface
├── assistant.py           # Main assistant logic
├── main.py               # Entry point
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
├── setup.py              # Package setup
└── README.md             # This file
```

## Commands

The assistant responds to various built-in commands:

### Information Commands
- **"What do you know about me?"** - Shows stored user information
- **"What time is it?"** - Current time
- **"What is the date?"** - Current date
- **"How long have we been talking?"** - Session duration

### Entertainment
- **"Tell me a joke"** - Random joke
- **"Fun fact"** - Interesting facts

### Memory & Personal
- **"My name is [name]"** - Saves your name
- **"I like [something]"** - Saves preferences
- **"I am [age] years old"** - Saves age

### System
- **"Goodbye/Exit/Stop"** - Shut down assistant
- **"Remind me to [task]"** - Set a reminder (basic)

## Configuration

Edit `config.py` to customize:

```python
@dataclass
class AssistantConfig:
    # Core settings
    assistant_name: str = "Hades"
    wake_word: str = "hades"
    
    # Audio settings (voice mode)
    command_record_seconds: int = 7
    wake_word_record_seconds: int = 2
    
    # AI model
    ollama_model: str = "qwen3:0.6b"
    
    # Memory
    memory_db_path: str = "assistant_memory.db"
    
    # Mode
    discrete_mode: bool = False
```

## Personality System

The assistant has four mood states:
- **Cheerful**: Upbeat and positive responses
- **Neutral**: Balanced, professional tone
- **Thoughtful**: Empathetic and considerate
- **Playful**: Casual with humor and wordplay

Moods change based on conversation context and user input.

## Memory System

The assistant remembers:
- User information (name, age, preferences)
- Conversation history
- Important facts and memories
- Session statistics

Memory is stored in a SQLite database and persists between sessions.

## Discrete Mode

Perfect for:
- Environments where voice isn't appropriate
- Testing and debugging
- Accessibility needs
- Server/remote deployments
This completes the modular refactoring of your voice assistant! The code is now:

1. **Organized into multiple files** for better maintainability
2. **Includes a discrete mode** that works without TTS/STT
3. **Fully configurable** through command-line arguments
4. **Well-documented** with a comprehensive README

You can now run it in either voice or text mode, customize the personality, and easily extend it with new features!

In discrete mode:
- No microphone or speaker required
- All interaction through text
- Same features as voice mode
- Emoji indicators for mood

## Extending the Assistant

### Adding New Commands

In `assistant.py`, add to the `process_command` method:

```python
elif "your command" in command_lower:
    response = "Your response"
    self.interface.output_response(response, "mood")
    return False
```

### Adding New Features

1. Create a new method in `features.py`:
```python
def your_feature(self, params):
    # Feature logic
    return result
```

2. Call it from `process_command` in `assistant.py`

### Custom Personalities

Modify personality traits in `config.py`:

```python
personality_traits: Dict[str, Any] = field(default_factory=lambda: {
    "core_traits": [
        "your custom traits",
        "another trait"
    ],
    "mood": "neutral",
    "formality": "casual",
    "enthusiasm_level": 0.7
})
```

## Troubleshooting

### Voice Mode Issues

**No audio input detected:**
- Check microphone permissions
- Verify PvRecorder device index
- Test with `--discrete` mode

**TTS not working:**
- Ensure Piper is installed
- Check model path in config
- Verify audio output device

### Ollama Connection Issues

**Model not found:**
```bash
ollama pull qwen3:0.6b
```

**Connection refused:**
- Start Ollama service: `ollama serve`
- Check if running: `ollama list`

### Memory Issues

**Database locked:**
- Close other instances
- Delete `.db-journal` files

**Reset memory:**
```bash
rm assistant_memory.db
```

## Performance Tips

1. **Use smaller models for faster responses:**
   - `qwen3:0.6b` - Fastest
   - `llama2:7b` - Balanced
   - `llama2:13b` - Best quality

2. **Adjust recording durations:**
   ```python
   command_record_seconds: int = 5  # Shorter for quick commands
   ```

3. **Limit conversation history:**
   ```python
   max_conversation_history: int = 10  # Reduce for less context
   ```

## Privacy

- All processing is done locally
- No data is sent to external servers
- Memory is stored in local SQLite database
- Voice is processed on-device

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. Feel free to use and modify.

## Acknowledgments

- Vosk for offline speech recognition
- Piper for local text-to-speech
- Ollama for local LLM inference
- The open-source AI community

## Future Enhancements

- [ ] Plugin system for easy extensibility
- [ ] Web interface dashboard
- [ ] Multi-language support
- [ ] Voice cloning for TTS
- [ ] Calendar integration
- [ ] Smart home control
- [ ] Custom wake word training
- [ ] Conversation export/import