import gradio as gr
import requests
import uvicorn
import os
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
CLOUD_URL = "https://ollama.com/api/chat"
LOCAL_URL = "http://127.0.0.1:11434/api/chat"

# Models
CLOUD_MODEL = "qwen2.5-coder:latest" 
LOCAL_MODEL = "jewelzufo/Qwen2.5-Coder-0.5B-Instruct-GGUF"

API_KEY = os.getenv("OLLAMA_API_KEY")

app = FastAPI(title="🤖 Qwen2.5")

def predict(message, history):

    messages = [{"role": "system", "content": "You are a professional coding assistant."}]
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})

    if API_KEY:
        try:
            print(f"☁️ Attempting Cloud Model ({CLOUD_MODEL})...")
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            payload = {"model": CLOUD_MODEL, "messages": messages, "stream": False}
            
            response = requests.post(CLOUD_URL, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            return response.json()['message']['content']
            
        except Exception as e:
            print(f"⚠️ Cloud failed: {e}. Switching to local...")

    try:
        print(f"🏠 Running Local Model ({LOCAL_MODEL})...")
        payload = {"model": LOCAL_MODEL, "messages": messages, "stream": False}
        response = requests.post(LOCAL_URL, json=payload, timeout=60)
        response.raise_for_status()
        return "*(Local)* " + response.json()['message']['content']
        
    except Exception as e:
        return f"❌ Critical Error: Both Cloud and Local failed. ({e})"

with gr.Blocks(theme=gr.themes.Soft()) as gui:
    gr.Markdown("# 🤖 Qwen2.5")
    status_label = "Cloud Preferred" if API_KEY else "Local Only (No API Key)"
    gr.Markdown(f"**Mode:** {status_label}")
    
    gr.ChatInterface(
        fn=predict,
        examples=["Write a Python script for a simple FastAPI server", "Explain recursion with a code example"],
    )

app = gr.mount_gradio_app(app, gui, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)