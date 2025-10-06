import os
import struct
import datetime
import random
import json
from vosk import Model, KaldiRecognizer
from pvrecorder import PvRecorder
import subprocess
import ollama
import re # --- NEW: Import the regular expression library ---

# --- Configuration ---
WAKE_WORD = "hades"
COMMAND_RECORD_SECONDS = 7 
WAKE_WORD_RECORD_SECONDS = 2
VOSK_MODEL_PATH = "models/vosk-model-en-in-0.5"

# --- Piper TTS Configuration ---
PIPER_MODEL_PATH = "models/en_US-lessac-medium.onnx" 
PIPER_SAMPLE_RATE = "22050"

# --- Ollama Configuration ---
# Updated to the model you are using
OLLAMA_MODEL = "qwen3:0.6b" 

# --- speak function (Unchanged) ---
def speak(text):
    """
    Speaks the given text using Piper TTS by streaming raw audio to aplay.
    """
    # Clean the text to avoid issues with shell commands
    cleaned_text = text.replace('"', "'").replace("`", "'").replace("$", "")
    print(f"Assistant: {cleaned_text}")
    
    command = (
        f'echo "{cleaned_text}" | '
        f'flatpak-spawn --host piper --model {PIPER_MODEL_PATH} --output-raw | '
        f'flatpak-spawn --host aplay -r {PIPER_SAMPLE_RATE} -f S16_LE -t raw -'
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


# --- NEW: Function to remove think tags from the response ---
def clean_response(text):
    """
    Removes <think>...</think> blocks from the text using regular expressions.
    """
    # The re.DOTALL flag allows the '.' to match newline characters
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Also remove any leading/trailing whitespace that might be left
    return cleaned_text.strip()


# --- MODIFIED: Function to interact with Ollama now cleans the response ---
def ask_ollama(prompt):
    """
    Sends a prompt to the Ollama model, cleans the response, and returns it.
    """
    try:
        print(f"User -> Ollama: {prompt}")
        speak("Thinking...") 
        
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            stream=False 
        )
        
        # Extract the raw response text
        raw_response_text = response['message']['content']
        
        # Clean the text to remove think tags before returning
        cleaned_response_text = clean_response(raw_response_text)
        
        return cleaned_response_text

    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        return "I'm having trouble connecting to my brain. Please make sure the Ollama service is running."


# --- process_command function (Unchanged) ---
def process_command(command):
    """
    Processes the transcribed command. Handles hard-coded commands first,
    then falls back to Ollama for everything else.
    """
    should_exit = False
    
    if "hello" in command:
        speak("Hello! How are you today?")
    elif "what time is it" in command:
        now = datetime.datetime.now().strftime("%I:%M %p")
        speak(f"The current time is {now}.")
    elif "what is the date" in command:
        today = datetime.datetime.now().strftime("%B %d, %Y")
        speak(f"Today's date is {today}.")
    elif "tell me a joke" in command:
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "What do you call fake spaghetti? An Impasta!"
        ]
        speak(random.choice(jokes))
    elif "goodbye" in command or "exit" in command or "stop" in command:
        speak("Goodbye! Shutting down.")
        should_exit = True
    else:
        ollama_response = ask_ollama(command)
        # Now speaks the cleaned response
        if ollama_response: # Only speak if there's something left to say
            speak(ollama_response)
    
    return should_exit

# --- transcribe_audio function (Unchanged) ---
def transcribe_audio(recorder, recognizer, duration):
    """Records audio for a set duration and transcribes it to text using Vosk offline."""
    try:
        recorder.start()
        print(f"Listening for {duration} seconds...")
        
        num_frames = int((recorder.sample_rate / recorder.frame_length) * duration)
        frames = []
        for _ in range(num_frames):
            frames.extend(recorder.read())
            
        print("Finished listening.")
        recorder.stop()

        frame_bytes = struct.pack("h" * len(frames), *frames)

        if recognizer.AcceptWaveform(frame_bytes):
            result = recognizer.Result()
        else:
            result = recognizer.PartialResult()

        text = json.loads(result).get("text", "").lower()
        return text if text else None

    except Exception as e:
        print(f"Error in transcription: {e}")
        return None

# --- main function (Unchanged) ---
def main():
    """Main function to run the voice assistant offline."""
    recorder = None
    
    try:
        if not os.path.exists(VOSK_MODEL_PATH):
            print(f"Vosk model not found at '{VOSK_MODEL_PATH}'.")
            return
            
        if not os.path.exists(PIPER_MODEL_PATH):
            print(f"Piper model not found at '{PIPER_MODEL_PATH}'.")
            return
        
        model = Model(VOSK_MODEL_PATH)
        recognizer = KaldiRecognizer(model, 16000)

        recorder = PvRecorder(device_index=-1, frame_length=512)
        
        speak("Hello, I am online. Say my name to give me a command.")
        
        while True:
            print(f"\nListening for wake word '{WAKE_WORD}'...")
            wake_word_text = transcribe_audio(recorder, recognizer, WAKE_WORD_RECORD_SECONDS)

            if wake_word_text and WAKE_WORD in wake_word_text:
                speak("Yes? How can I help?")
                
                command = transcribe_audio(recorder, recognizer, COMMAND_RECORD_SECONDS)
                
                if command:
                    print(f"You said: {command}")
                    if process_command(command):
                        break
                else:
                    speak("I didn't catch that. Please try again.")

    except KeyboardInterrupt:
        print("\nShutting down.")
    except IndexError:
        print("No audio devices found. Please check your microphone connection.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if recorder is not None:
            recorder.delete()

if __name__ == '__main__':
    main()