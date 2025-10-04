import os
import struct
import json
import subprocess
from typing import Optional
from vosk import Model, KaldiRecognizer
from pvrecorder import PvRecorder
from .base import AssistantInterface
from config import AssistantConfig

class VoiceInterface(AssistantInterface):
    """Voice-based interface using STT and TTS"""
    
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.model = None
        self.recognizer = None
        self.recorder = None
        
    def initialize(self) -> None:
        """Initialize voice components"""
        if not os.path.exists(self.config.vosk_model_path):
            raise FileNotFoundError(f"Vosk model not found at '{self.config.vosk_model_path}'")
        
        if not os.path.exists(self.config.piper_model_path):
            raise FileNotFoundError(f"Piper model not found at '{self.config.piper_model_path}'")
        
        self.model = Model(self.config.vosk_model_path)
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.recorder = PvRecorder(device_index=-1, frame_length=512)
        
    def get_user_input(self, prompt: Optional[str] = None) -> Optional[str]:
        """Get voice input from user"""
        if prompt == "wake_word":
            return self._transcribe_audio(self.config.wake_word_record_seconds)
        else:
            return self._transcribe_audio(self.config.command_record_seconds)
    
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
        return text and self.config.wake_word in text.lower()
    
    def cleanup(self) -> None:
        """Clean up voice resources"""
        if self.recorder:
            self.recorder.delete()