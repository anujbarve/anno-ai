import wave
import struct
from pvrecorder import PvRecorder

def record_audio(output_file_path, record_seconds=5):
    """
    Records audio from the default microphone for a specified duration
    and saves it to a WAV file.

    :param output_file_path: Path to save the output WAV file.
    :param record_seconds: Duration of the recording in seconds.
    """
    recorder = None
    wav_file = None
    try:
        # -1 indicates the default audio device.
        recorder = PvRecorder(device_index=-1, frame_length=512)
        
        # Get the audio format details from the recorder
        sample_rate = recorder.sample_rate
        num_channels = 1  # pvrecorder records in mono
        sample_width = 2  # 16-bit audio

        # Open a WAV file for writing
        wav_file = wave.open(output_file_path, 'wb')
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)

        print(f"Recording for {record_seconds} seconds...")
        recorder.start()

        # Calculate the total number of frames to read
        total_frames_to_read = int((sample_rate / recorder.frame_length) * record_seconds)

        for _ in range(total_frames_to_read):
            # Read a frame of audio
            frame = recorder.read()
            # Pack the audio frame into bytes and write to the file
            wav_file.writeframes(struct.pack("h" * len(frame), *frame))

        print(f"Recording finished. Audio saved to {output_file_path}")

    except IndexError:
        print(f"No audio devices found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up resources
        if recorder is not None:
            recorder.stop()
            recorder.delete()
        
        if wav_file is not None:
            wav_file.close()


if __name__ == '__main__':
    # List available audio devices
    devices = PvRecorder.get_available_devices()
    print("Available audio devices:")
    for i, device in enumerate(devices):
        print(f"  [{i}] {device}")

    # Start the recording process
    record_audio('output.wav', record_seconds=5)
