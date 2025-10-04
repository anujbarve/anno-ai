#!/usr/bin/env python3
"""
Train a personalized wake word model for the voice assistant
"""

import argparse
import os
from voice_training import WakeWordTrainer

def main():
    parser = argparse.ArgumentParser(
        description="Train a personalized wake word model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train default wake word
  python train_wake_word.py
  
  # Train custom wake word
  python train_wake_word.py --wake-word "jarvis"
  
  # Train with more samples
  python train_wake_word.py --positive-samples 30 --negative-samples 30
        """
    )
    
    parser.add_argument('--wake-word', type=str, default='hades',
                      help='Wake word to train (default: hades)')
    parser.add_argument('--positive-samples', type=int, default=20,
                      help='Number of positive samples to collect (default: 20)')
    parser.add_argument('--negative-samples', type=int, default=20,
                      help='Number of negative samples to collect (default: 20)')
    parser.add_argument('--test-only', action='store_true',
                      help='Test existing model without training')
    
    args = parser.parse_args()
    
    trainer = WakeWordTrainer(args.wake_word)
    
    try:
        if args.test_only:
            # Load and test existing model
            model_path = f"models/wake_word_{args.wake_word}.pkl"
            if os.path.exists(model_path):
                import pickle
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                print(f"✅ Loaded model for '{args.wake_word}'")
                trainer.test_model(model_data)
            else:
                print(f"❌ No trained model found for '{args.wake_word}'")
                print(f"   Run without --test-only to train a new model")
        else:
            # Collect samples
            positive_samples, negative_samples = trainer.collect_samples(
                num_positive=args.positive_samples,
                num_negative=args.negative_samples
            )
            
            # Train model
            if positive_samples and negative_samples:
                model_data = trainer.train_model(positive_samples, negative_samples)
                
                # Test model
                print("\n" + "="*50)
                print("Would you like to test the model now? (recommended)")
                if input("Test model? (y/n): ").lower() == 'y':
                    trainer.test_model(model_data)
            else:
                print("❌ Training cancelled - insufficient samples collected")
                
    except KeyboardInterrupt:
        print("\n\n⚡ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        trainer.cleanup()
        print("\n✅ Cleanup complete")


if __name__ == "__main__":
    main()