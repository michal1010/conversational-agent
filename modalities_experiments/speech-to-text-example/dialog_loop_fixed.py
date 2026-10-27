import json
import requests
import os
from datetime import datetime
from RealtimeSTT import AudioToTextRecorder

OLLAMA_URL = "http://localhost:11434/api/generate"

system_prompt = """
You are an empathetic conversational agent. Your goal is to establish "common ground" with the user.
The user is going to tell you about an emotional event. 
Use the "explicit confirmation" strategy: acknowledge their feelings, and ask a gentle clarification question to explore the event further.
Keep your response strictly under 3 sentences. Be warm and conversational.
"""

if __name__ == '__main__':
    print("Loading Whisper model (base.en)...")
    # 💡 Try changing "base.en" to "small.en" if you want even better accent accuracy!
    recorder = AudioToTextRecorder(model="small.en", language="en")
    
    # 💡 NEW: Initialize an empty list to hold the entire session's history
    conversation_history = []
    
    # 💡 NEW: Create a unique filename for this session based on the current time
    session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"dialog_session_{session_time}.json"
    
    print("\n" + "="*60)
    print("🚀 FULL DIALOG SYSTEM ACTIVE (VOICE TRIGGER MODE)")
    print(f"📁 Transcripts will be saved to: {log_filename}")
    print("1. Talk naturally. The system will capture your pauses.")
    print("2. When you are COMPLETELY done, just say: 'I am done.' or 'That is all.'")
    print("3. Press Ctrl+C only when you want to quit the entire program.")
    print("="*60 + "\n")

    try:
        while True:
            full_turn_chunks = []
            print("\n🎤 [Your Turn] Start speaking...")
            
            while True:
                text = recorder.text()
                if text:
                    print(f"  -> {text}")
                    full_turn_chunks.append(text)
                    
                    text_lower = text.lower()
                    clean_text = text_lower.replace(".", "").replace(",", "").strip()
                    
                    if clean_text in ["i am done", "i'm done", "that is all", "that's all", "over"]:
                        break

            complete_turn_string = " ".join(full_turn_chunks)
            print("\n✅ Turn complete! Sending to Dialog Manager...")
            
            # 💡 NEW: Package the USER'S turn and append to history
            user_payload = {
                "role": "user",
                "timestamp": datetime.now().isoformat(),
                "content": complete_turn_string
            }
            conversation_history.append(user_payload)
            
            # 💡 NEW: Save the updated history to the JSON file immediately
            with open(log_filename, "w", encoding="utf-8") as f:
                json.dump(conversation_history, f, indent=4)

            print("🧠 Agent is thinking...")
            
            prompt_text = f"{system_prompt}\n\nUser: {complete_turn_string}\n\nAgent:"
            
            llm_payload = {
                "model": "llama3",
                "prompt": prompt_text,
                "stream": False 
            }
            
            try:
                response = requests.post(OLLAMA_URL, json=llm_payload)
                agent_reply = response.json().get("response", "Error generating response.")
                
                print(f"\n🤖 Agent: {agent_reply}\n")
                print("-" * 60)
                
                # 💡 NEW: Package the AGENT'S turn and append to history
                agent_payload = {
                    "role": "agent",
                    "timestamp": datetime.now().isoformat(),
                    "content": agent_reply
                }
                conversation_history.append(agent_payload)
                
                # 💡 NEW: Save the updated history to the JSON file again
                with open(log_filename, "w", encoding="utf-8") as f:
                    json.dump(conversation_history, f, indent=4)
                
            except requests.exceptions.ConnectionError:
                print("\n❌ Error: Could not connect to Ollama.")
                break

    except KeyboardInterrupt:
        print(f"\n\n🛑 Shutting down dialog system safely.")
        print(f"📁 Final conversation saved to: {log_filename}")
        recorder.shutdown()