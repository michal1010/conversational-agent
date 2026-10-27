import sounddevice as sd
import soundfile as sf
import queue
import sys

# We force 16000 Hz because AI models (like Audeering/Whisper) are trained on 16kHz audio
SAMPLE_RATE = 16000
CHANNELS = 1

q = queue.Queue()

def callback(indata, frames, time, status):
    """Pushes the raw audio data from your M4 Pro mic into our queue."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

def record_audio(filename="test_audio.wav"):
    try:
        # Check if the mic is available
        device_info = sd.query_devices(kind='input')
        print(f"🎤 Using Microphone: {device_info['name']}")
        
        input("Press [ENTER] to start recording...")
        print("\n🔴 Recording... (Press Ctrl+C to stop)")
        
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback):
            audio_data = []
            while True:
                audio_data.append(q.get())
                
    except KeyboardInterrupt:
        print("\n\n🛑 Recording stopped. Saving file...")
        
        # Flatten the list of audio chunks into one continuous array
        import numpy as np
        full_audio = np.concatenate(audio_data, axis=0)
        
        # Save it to a .wav file
        sf.write(filename, full_audio, SAMPLE_RATE)
        print(f"✅ Successfully saved to: {filename}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    record_audio("my_voice_test.wav")