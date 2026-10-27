import json
from datetime import datetime
from RealtimeSTT import AudioToTextRecorder

if __name__ == '__main__':
    print("Loading Whisper model...")
    # 💡 Try changing "base.en" to "small.en" if you want even better accent accuracy!
    recorder = AudioToTextRecorder(model="small.en", language="en") # for lighter model use base.en
    
    print("\n" + "="*50)
    print("🚀 Ready! Start reading your paragraph.")
    print("Take natural pauses. Press Ctrl+C when your turn is completely finished.")
    print("="*50 + "\n")

    # We will store every chunk here
    full_turn_chunks = []

    try:
        while True:
            text = recorder.text()
            if text:
                print(f"User: {text}")
                full_turn_chunks.append(text)
                
    except KeyboardInterrupt:
        print("\n\n🛑 Turn ended! Processing for LLM...")
        recorder.shutdown()
        
        # 1. Stitch the list of chunks into one giant string
        complete_turn_string = " ".join(full_turn_chunks)
        
        # 2. Structure it exactly how an LLM (like Llama or OpenAI) expects it
        llm_payload = {
            "role": "user",
            "timestamp": datetime.now().isoformat(),
            "content": complete_turn_string
        }
        
        # 3. Save it to a JSON file
        with open("user_turn.json", "w", encoding="utf-8") as json_file:
            json.dump(llm_payload, json_file, indent=4)
            
        print("✅ Successfully saved to 'user_turn.json'!")
        print("Here is exactly what will be fed to your Dialog Manager:\n")
        print(json.dumps(llm_payload, indent=4))