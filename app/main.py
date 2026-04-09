import gradio as gr
import requests
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "jewelzufo/Qwen2.5-Coder-0.5B-Instruct-GGUF"

app = FastAPI(title="Pro Coding Assistant")

def predict(message, history):
    messages = [
        {"role": "system", "content": "You are a specialized coding assistant. Provide clean, efficient, and well-documented code."}
    ]
    
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    
    messages.append({"role": "user", "content": message})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.7}
    }

    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()['message']['content']
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# --- UI DESIGN ---
with gr.Blocks(theme=gr.themes.Soft()) as gui:
    gr.Markdown("# 🤖 Qwen2.5 Local Coder")
    
    gr.ChatInterface(
        fn=predict,
        examples=["Write a Python script for a simple FastAPI server", "Explain recursion with a code example"],
    )

app = gr.mount_gradio_app(app, gui, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)