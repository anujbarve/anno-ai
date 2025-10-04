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