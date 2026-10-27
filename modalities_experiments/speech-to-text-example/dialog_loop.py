import json
import requests
from datetime import datetime
from RealtimeSTT import AudioToTextRecorder

# The endpoint for your local Ollama server
OLLAMA_URL = "http://localhost:11434/api/generate"

# The psychological framework for your agent
system_prompt = """
You are an empathetic conversational agent. Your goal is to establish "common ground" with the user.
The user is going to tell you about an emotional event. 
Use the "explicit confirmation" strategy: acknowledge their feelings, and ask a gentle clarification question to explore the event further.
Keep your response strictly under 3 sentences. Be warm and conversational.
"""

if __name__ == '__main__':
    print("Loading Whisper model (base.en)...")
    recorder = AudioToTextRecorder(model="base.en", language="en")
    
    print("\n" + "="*60)
    print("🚀 FULL DIALOG SYSTEM ACTIVE")
    print("1. Talk naturally. The system will capture your pauses.")
    print("2. When you are COMPLETELY done with your turn, press Ctrl+C.")
    print("3. To exit the program entirely, press Ctrl+C before speaking.")
    print("="*60 + "\n")

    # The main conversation loop
    while True:
        full_turn_chunks = []
        print("\n🎤 [Your Turn] Start speaking...")
        
        try:
            # The inner loop: constantly listens and accumulates your speech
            while True:
                text = recorder.text()
                if text:
                    print(f"  -> {text}")
                    full_turn_chunks.append(text)
                    
        except KeyboardInterrupt:
            # You pressed Ctrl+C! This breaks the listening loop.
            
            # If you pressed it without saying anything, exit the program.
            if not full_turn_chunks:
                print("\n\n🛑 Shutting down dialog system.")
                recorder.shutdown()
                break
                
            # Otherwise, package your turn and send it to the LLM
            complete_turn_string = " ".join(full_turn_chunks)
            print("\n✅ Turn complete! Sending to Dialog Manager...")
            print("🧠 Agent is thinking...")
            
            prompt_text = f"{system_prompt}\n\nUser: {complete_turn_string}\n\nAgent:"
            
            payload = {
                "model": "llama3",
                "prompt": prompt_text,
                "stream": False # We wait for the full response to print it at once
            }
            
            try:
                # Call the local LLM
                response = requests.post(OLLAMA_URL, json=payload)
                agent_reply = response.json().get("response", "Error generating response.")
                
                print(f"\n🤖 Agent: {agent_reply}\n")
                print("-" * 60)
                
            except requests.exceptions.ConnectionError:
                print("\n❌ Error: Could not connect to Ollama.")
                print("Did you forget to run 'ollama run llama3' in a separate terminal?")
                break