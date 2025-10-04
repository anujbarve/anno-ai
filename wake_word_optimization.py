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