# Enhanced Voice Assistant v2.0

A powerful, modular voice assistant with personality, memory, and multiple interfaces. Features personalized wake word detection, local AI processing, and both voice and text interaction modes.

## 🌟 Key Features

### Core Capabilities
- **🎙️ Dual Interface**: Voice mode with STT/TTS or discrete text-only mode
- **🧠 Personality System**: Dynamic mood and personality traits
- **💾 Persistent Memory**: Remembers users, conversations, and preferences
- **🔒 Privacy-First**: All processing done locally with Ollama
- **🎯 Personalized Wake Word**: Train the assistant to recognize YOUR voice
- **🔌 Modular Architecture**: Easy to extend and customize

### Assistant Features
- Natural conversation with context awareness
- User information tracking (name, preferences, etc.)
- Mood-based responses (cheerful, thoughtful, playful, neutral)
- Built-in commands and utilities
- Reminder system (basic)
- Fun facts and jokes
- Session tracking and statistics

## 📋 Prerequisites

- Python 3.8+
- Microphone (for voice mode)
- Ollama installed and running
- Required models (see Installation)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd enhanced-voice-assistant
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Required Models

#### For Voice Mode:
1. **Vosk Model** (Speech Recognition):
   ```bash
   mkdir -p models
   cd models
   wget https://alphacephei.com/vosk/models/vosk-model-en-in-0.5.zip
   unzip vosk-model-en-in-0.5.zip
   ```

2. **Piper Model** (Text-to-Speech):
   ```bash
   # Download Piper TTS model
   wget https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-lessac-medium.onnx
   ```

#### For All Modes:
3. **Ollama Model**:
   ```bash
   # Install Ollama if not already installed
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull the AI model
   ollama pull qwen3:0.6b
   ```

### 4. Install Additional Dependencies for Wake Word Training
```bash
pip install librosa soundfile scikit-learn
```

## 📁 Project Structure

```
enhanced-voice-assistant/
├── config.py                 # Configuration settings
├── memory.py                 # Memory and conversation management
├── personality.py            # Personality engine
├── features.py              # Additional features
├── assistant.py             # Main assistant logic
├── utils.py                 # Utility functions
├── main.py                  # Entry point
├── voice_training.py        # Wake word training system
├── train_wake_word.py       # Wake word training script
├── wake_word_optimization.py # Advanced wake word features
├── interfaces/              # Interface implementations
│   ├── __init__.py
│   ├── base.py             # Abstract interface
│   ├── voice_interface.py  # Voice STT/TTS interface
│   └── text_interface.py   # Text-only interface
├── models/                  # Model files directory
│   ├── vosk-model-en-in-0.5/
│   ├── en_US-lessac-medium.onnx
│   └── wake_word_*.pkl     # Personalized wake word models
├── voice_samples/           # Voice training samples
├── requirements.txt         # Python dependencies
├── setup.py                # Package setup
└── README.md               # This file
```

## 🎯 Personalized Wake Word Training

Train the assistant to recognize your specific voice for better wake word detection:

### Quick Training (Recommended)
```bash
# Train with default settings (20 positive, 20 negative samples)
python train_wake_word.py --wake-word "hades"
```

### Advanced Training
```bash
# More samples for better accuracy
python train_wake_word.py --wake-word "jarvis" --positive-samples 30 --negative-samples 30

# Test existing model
python train_wake_word.py --wake-word "hades" --test-only
```

### Training Process:
1. **Positive Samples**: Say your wake word clearly 20+ times
2. **Negative Samples**: Say other random words 20+ times
3. **Testing**: Test the model with real-time detection
4. The model is automatically saved and loaded when you run the assistant

### Training Tips:
- Speak in your normal voice
- Vary your tone slightly
- Train in your typical usage environment
- Include some background noise for robustness
- Use similar-sounding words in negative samples

## 🎮 Usage

### Voice Mode (Default)
```bash
python main.py
```
- Say the wake word (default: "hades") to activate
- Speak your command
- Say "goodbye" to exit

### Text Mode (Discrete)
```bash
python main.py --discrete
```
- Type your messages
- No microphone or speakers required
- Perfect for quiet environments or accessibility

### Custom Configuration
```bash
# Different wake word
python main.py --wake-word "jarvis" --name "Jarvis"

# Different AI model
python main.py --model "llama2:7b"

# Custom database
python main.py --memory-db "my_assistant.db"

# Combine options
python main.py --discrete --name "Assistant" --model "llama2:13b"
```

## 💬 Commands and Features

### Basic Commands
| Command | Description |
|---------|-------------|
| "Hello" | Greeting |
| "What time is it?" | Current time |
| "What's the date?" | Today's date |
| "Tell me a joke" | Random joke |
| "Fun fact" | Interesting fact |
| "Goodbye/Exit/Stop" | Shut down |

### Memory Commands
| Command | Description |
|---------|-------------|
| "What do you know about me?" | Show stored information |
| "My name is [name]" | Save your name |
| "I like [something]" | Save preferences |
| "I am [age] years old" | Save age |

### Session Commands
| Command | Description |
|---------|-------------|
| "How long have we been talking?" | Session duration |
| "Remind me to [task]" | Set a reminder |

### Everything Else
Any other input is processed by the AI model for natural conversation!

## ⚙️ Configuration

Edit `config.py` to customize:

```python
@dataclass
class AssistantConfig:
    # Basic settings
    assistant_name: str = "Hades"
    wake_word: str = "hades"
    
    # Voice settings
    command_record_seconds: int = 7
    wake_word_record_seconds: int = 2
    
    # AI model
    ollama_model: str = "qwen3:0.6b"
    
    # Personality
    personality_traits: Dict[str, Any] = ...
    
    # Mode
    discrete_mode: bool = False
```

## 🧠 Personality System

The assistant has four mood states that affect responses:

| Mood | Triggered By | Response Style |
|------|--------------|----------------|
| **Cheerful** 😊 | Positive words (thank, great, love) | Upbeat and positive |
| **Neutral** 😐 | Default state | Balanced and professional |
| **Thoughtful** 🤔 | Negative emotions (sad, upset) | Empathetic and considerate |
| **Playful** 😄 | Fun words (joke, play, game) | Casual with humor |

## 💾 Memory System

The assistant maintains several types of memory:

1. **User Information**: Name, age, preferences
2. **Conversation History**: Recent exchanges with context
3. **Long-term Memories**: Important facts and information
4. **Session Data**: Usage statistics and patterns

Memory persists between sessions in a SQLite database.

## 🔧 Troubleshooting

### Voice Mode Issues

**Problem: Wake word not detected reliably**
- Train a personalized wake word model
- Adjust microphone sensitivity
- Check for background noise
- Use `--discrete` mode as alternative

**Problem: No audio output**
- Check speaker/headphone connection
- Verify Piper TTS installation
- Test with: `echo "test" | piper --model models/en_US-lessac-medium.onnx --output-raw | aplay -r 22050 -f S16_LE -t raw -`

**Problem: Speech not recognized**
- Check microphone permissions
- Test microphone with system recorder
- Verify Vosk model is downloaded
- Speak clearly and not too fast

### Ollama Issues

**Problem: Connection refused**
```bash
# Start Ollama service
ollama serve

# Verify it's running
ollama list
```

**Problem: Model not found**
```bash
# Pull the model
ollama pull qwen3:0.6b

# Or try a different model
ollama pull llama2:7b
python main.py --model "llama2:7b"
```

### General Issues

**Problem: Dependencies missing**
```bash
pip install -r requirements.txt --upgrade
```

**Problem: Database locked**
- Close other instances
- Delete `.db-journal` files
- Worst case: `rm assistant_memory.db` (loses memory)

## 🚀 Performance Optimization (continued)

### For Faster Responses:
1. **Use smaller models**:
   - `qwen3:0.6b` - Fastest
   - `llama2:7b` - Balanced
   - `llama2:13b` - Best quality

2. **Optimize recording duration**:
   ```python
   # In config.py
   command_record_seconds: int = 5  # Shorter for quick commands
   wake_word_record_seconds: int = 1.5  # Faster wake detection
   ```

3. **Reduce conversation history**:
   ```python
   max_conversation_history: int = 10  # Less context = faster
   ```

4. **Use GPU acceleration** (if available):
   ```bash
   # Check Ollama GPU support
   ollama run qwen3:0.6b --verbose
   ```

### For Better Accuracy:
1. **Train personalized wake word** with more samples
2. **Use larger Vosk models** for better STT
3. **Increase recording duration** for complex commands
4. **Fine-tune confidence thresholds**

## 🧪 Testing

### Run Unit Tests
```bash
# Test memory system
python main.py --test-memory

# Test personality engine
python main.py --test-personality

# Test wake word detection
python train_wake_word.py --wake-word "hades" --test-only
```

### Manual Testing Checklist
- [ ] Wake word detection in quiet environment
- [ ] Wake word detection with background noise
- [ ] Various commands (time, date, jokes)
- [ ] Memory persistence across sessions
- [ ] Mood changes based on conversation
- [ ] Error handling (no mic, no Ollama)
- [ ] Discrete mode functionality

## 🔌 Extending the Assistant

### Adding New Commands

1. Edit `assistant.py`, find the `process_command` method:
```python
elif "weather" in command_lower:
    # Add weather functionality
    response = self.get_weather()  # Implement this
    self.interface.output_response(response, "neutral")
    return False
```

2. For complex features, add to `features.py`:
```python
def get_weather(self, location: str = None) -> str:
    # Implement weather API integration
    return f"Weather information for {location}"
```

### Adding New Personalities

Edit personality traits in `config.py`:
```python
personality_traits: Dict[str, Any] = field(default_factory=lambda: {
    "core_traits": [
        "professional and formal",
        "technical and precise",
        "avoids humor"
    ],
    "mood": "neutral",
    "formality": "formal",
})
```

### Creating Plugins

Create a new file in `plugins/`:
```python
# plugins/weather_plugin.py
class WeatherPlugin:
    def __init__(self, assistant):
        self.assistant = assistant
    
    def process(self, command: str) -> tuple[bool, str]:
        if "weather" in command:
            return True, "Weather is sunny!"
        return False, None
```

## 📊 Statistics and Monitoring

View assistant usage statistics:
```python
# In the assistant, say:
"How long have we been talking?"
"What do you know about me?"
```

Access raw statistics programmatically:
```python
from memory import ConversationMemory

memory = ConversationMemory("assistant_memory.db")
stats = memory.get_recent_conversations(10)
```

## 🐳 Docker Support (Coming Soon)

```dockerfile
# Dockerfile (planned)
FROM python:3.9-slim
# ... Docker configuration
```

## 🌐 API Integrations

While the assistant runs fully offline, you can add online features:

### Weather API Example
```python
# In features.py
import requests

def get_weather(self, city: str) -> str:
    # Add your API key
    response = requests.get(f"https://api.weather.com/...")
    return format_weather(response.json())
```

### Home Automation
```python
# Integration with Home Assistant, OpenHAB, etc.
def control_lights(self, command: str) -> str:
    # Implement smart home control
    pass
```

## 🔐 Privacy & Security

- **100% Local Processing**: No data sent to external servers
- **Offline Capable**: Works without internet (except Ollama model downloads)
- **Local Storage**: All memory stored in local SQLite database
- **No Telemetry**: No usage tracking or analytics
- **Open Source**: Fully auditable code

### Data Storage Locations
- `assistant_memory.db` - Conversation history and user data
- `models/` - AI models and voice models
- `voice_samples/` - Training recordings (can be deleted after training)

## 🤝 Contributing

We welcome contributions! Here's how to help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Contribution Ideas
- New personality traits
- Additional commands
- Language support
- Plugin system
- Web interface
- Mobile app
- Better wake word detection
- Voice cloning for TTS

## 📝 Development Guidelines

### Code Style
- Use Python 3.8+ features
- Follow PEP 8
- Add type hints
- Document functions
- Keep functions focused

### Adding Features
1. Update relevant module
2. Add tests if applicable
3. Update README
4. Test in both voice and text modes

## 🐛 Known Issues

1. **Wake word detection** varies by accent and microphone
2. **TTS emotion** is limited by Piper capabilities
3. **Background noise** affects recognition accuracy
4. **Memory search** is basic keyword matching

## 🗺️ Roadmap

### Version 2.1
- [ ] Plugin system architecture
- [ ] Web dashboard
- [ ] Multi-user support
- [ ] Voice profiles

### Version 2.2
- [ ] Multi-language support
- [ ] Advanced memory search
- [ ] Calendar integration
- [ ] Email capabilities

### Version 3.0
- [ ] Neural voice cloning
- [ ] Vision capabilities
- [ ] Proactive suggestions
- [ ] Learning from corrections

## 📚 Resources

### Models
- [Vosk Models](https://alphacephei.com/vosk/models)
- [Piper TTS](https://github.com/rhasspy/piper)
- [Ollama Models](https://ollama.ai/library)

### Documentation
- [Vosk API](https://alphacephei.com/vosk/api)
- [Ollama Docs](https://github.com/jmorganca/ollama)
- [Librosa](https://librosa.org/doc/latest/index.html)

### Community
- Report issues on GitHub
- Join discussions
- Share your modifications

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Vosk** - Offline speech recognition
- **Piper** - Local text-to-speech  
- **Ollama** - Local LLM inference
- **Picovoice** - Wake word detection research
- **Open Source Community** - For amazing tools and libraries

## 💡 Tips & Tricks

### For Best Experience:
1. **Train your wake word** in your normal environment
2. **Use a good microphone** for better recognition
3. **Keep sessions focused** for better context
4. **Experiment with moods** using different phrases
5. **Regular cleanup** of old memory data

### Hidden Features:
- The assistant learns your preferences over time
- Mood affects joke selection
- Multiple ways to phrase commands
- Context carries across conversations

## ❓ FAQ

**Q: Can I use this without a microphone?**
A: Yes! Use `--discrete` mode for text-only interaction.

**Q: How do I change the voice?**
A: Download different Piper models and update the path in config.py.

**Q: Can multiple people use it?**
A: Currently single-user, but stores different names if mentioned.

**Q: Does it work offline?**
A: Yes, completely offline after initial setup.

**Q: How much disk space needed?**
A: ~2GB for models, plus memory database growth (~1MB per month of use).

---

Made with ❤️ by the open source community. Enjoy your new AI assistant!