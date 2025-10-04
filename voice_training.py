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
            if audio_data:
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
            if audio_data:
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
                
                if audio_data:
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