import sys
import queue
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer

q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

print("Loading Vosk model (en-us)...")
model = Model(lang="en-us")

# 💡 STOLEN FROM OFFICIAL SCRIPT: Dynamically get your Mac's preferred samplerate
device_info = sd.query_devices(None, "input")
samplerate = int(device_info["default_samplerate"])
print(f"Detected Mac microphone samplerate: {samplerate} Hz")

recognizer = KaldiRecognizer(model, samplerate)

# This list holds the finalized sentences across your pauses
full_turn_words = []

print("\n" + "="*60)
print("🎤 START SPEAKING. Take pauses naturally.")
print("Press Ctrl+C when you are completely finished with the story.")
print("="*60 + "\n")

try:
    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, 
                           dtype="int16", channels=1, callback=callback):
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                # You paused. Vosk finalized a chunk. Parse the JSON and save the text!
                result_dict = json.loads(recognizer.Result())
                text = result_dict.get("text", "")
                if text:
                    full_turn_words.append(text)
            else:
                # You are actively speaking. Update the live stream.
                partial_dict = json.loads(recognizer.PartialResult())
                current_partial = partial_dict.get("partial", "")
                
                # Stitch past chunks + current live speech
                display_text = " ".join(full_turn_words) + " " + current_partial
                # The :<100 pads with spaces to overwrite old text cleanly on Mac terminal
                print(f"User: {display_text.strip():<100}", end="\r")
                
except KeyboardInterrupt:
    # 💡 When you press Ctrl+C, grab any lingering words left in the buffer
    final_result = json.loads(recognizer.FinalResult())
    last_text = final_result.get("text", "")
    if last_text:
        full_turn_words.append(last_text)
        
    final_turn_string = " ".join(full_turn_words)
    print(f"\n\n✅ [FINAL COMPLETE TURN READY FOR LLM]:\n'{final_turn_string}'\n")
except Exception as e:
    print(f"\nError: {e}")