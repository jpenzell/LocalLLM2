import gradio as gr
import requests
from typing import List, Dict
import logging
from pydantic import ConfigDict

# Configure Pydantic
model_config = ConfigDict(arbitrary_types_allowed=True)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_all_models() -> List[Dict[str, str]]:
    """Fetch all available models from Ollama"""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        response.raise_for_status()
        data = response.json()
        if 'models' in data:
            return [{
                'name': model['name'],
                'size': model.get('size', 'N/A'),
                'modified_at': model.get('modified_at', 'N/A'),
                'digest': model.get('digest', 'N/A')
            } for model in data['models']]
        return []
    except Exception as e:
        logging.error(f"Error fetching all models: {e}")
        return []

def get_model_info(model_name: str) -> str:
    """Get information about a specific model"""
    try:
        response = requests.post('http://localhost:11434/api/show', json={"name": model_name})
        response.raise_for_status()
        info = response.json()
        return (
            f"Model: {model_name}\n"
            f"Size: {info.get('size', 'N/A')}\n"
            f"Format: {info.get('format', 'N/A')}\n"
            f"Family: {info.get('family', 'N/A')}\n"
            f"Parameter Size: {info.get('parameter_size', 'N/A')}\n"
            f"Quantization Level: {info.get('quantization_level', 'N/A')}"
        )
    except Exception as e:
        logging.error(f"Error fetching model info: {e}")
        return f"Error: Could not fetch model information - {str(e)}"

def generate_response(prompt, history, system_prompt, temperature, model):
    """Send a request to Ollama and get the response"""
    try:
        # Construct context from history
        context = ""
        for msg in history:
            context += f"User: {msg[0]}\nAssistant: {msg[1]}\n"
        
        # Add current prompt
        full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nUser: {prompt}\nAssistant:"
        
        response = requests.post('http://localhost:11434/api/generate',
                               json={
                                   "model": model,
                                   "prompt": full_prompt,
                                   "temperature": float(temperature),
                                   "stream": False
                               })
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def chat_interface(message: str, 
                  history: List[List[str]], 
                  system_prompt: str,
                  temperature: float,
                  model: str):
    if not message.strip():  # Skip empty messages
        return history, history, ""
    
    # Generate response with selected model
    response = generate_response(message, history, system_prompt, temperature, model)
    
    # Update history
    history = history or []
    history.append([message, response])
    
    return history, history, ""

def get_available_models() -> List[str]:
    """Get list of available models from Ollama"""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        response.raise_for_status()
        data = response.json()
        models = []
        if 'models' in data:
            models = [model['name'] for model in data['models']]
        if not models:
            models = ["mistral"]
        return models
    except Exception as e:
        logging.error(f"Error fetching available models: {e}")
        return ["mistral"]

def pull_model(model_name: str) -> str:
    """Pull a new model from Ollama"""
    try:
        response = requests.post('http://localhost:11434/api/pull', json={"name": model_name})
        response.raise_for_status()
        return f"Successfully pulled model: {model_name}"
    except Exception as e:
        logging.error(f"Error pulling model {model_name}: {e}")
        return f"Error: {str(e)}"

def delete_model(model_name: str) -> str:
    """Delete a model from Ollama"""
    try:
        response = requests.delete(f'http://localhost:11434/api/delete', json={"name": model_name})
        response.raise_for_status()
        return f"Successfully deleted model: {model_name}"
    except Exception as e:
        logging.error(f"Error deleting model {model_name}: {e}")
        return f"Error: {str(e)}"

def update_models_table():
    """Update the models table with current data"""
    models = fetch_all_models()
    if models:
        data = [[m['name'], m['size'], m['modified_at'], m['digest']] for m in models]
        return data
    return []

def update_model_selector():
    """Update the model selector dropdown with current models"""
    models = get_available_models()
    return gr.update(choices=models, value=models[0] if models else "mistral")

# Create the Gradio interface
def create_gradio_interface():
    with gr.Blocks() as iface:
        gr.Markdown("# Ollama Chat Interface")
        
        with gr.Tabs():
            with gr.TabItem("Chat"):
                # Chat interface components
                with gr.Row():
                    with gr.Column(scale=2):
                        available_models = get_available_models()
                        model_selector = gr.Dropdown(
                            label="Select AI Model",
                            choices=available_models,
                            value=available_models[0] if available_models else "mistral",
                            interactive=True,
                            allow_custom_value=True
                        )
                    with gr.Column(scale=1):
                        refresh_models = gr.Button("🔄 Refresh Models")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(height=300)
                        msg = gr.Textbox(
                            label="Type your message here",
                            lines=1,
                            autofocus=True
                        )
                        with gr.Row():
                            clear = gr.Button("Clear")
                            submit = gr.Button("Submit", variant="primary")
                    
                    with gr.Column(scale=1):
                        system_prompt = gr.Textbox(
                            label="System Prompt",
                            value="You are a helpful AI assistant.",
                            lines=2
                        )
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature"
                        )
            
            with gr.TabItem("Model Management"):
                gr.Markdown("## Manage Ollama Models")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        new_model_name = gr.Textbox(
                            label="Model to Install",
                            placeholder="Enter model name (e.g., llama2, mistral)"
                        )
                    with gr.Column(scale=1):
                        pull_model_btn = gr.Button("Install Model", variant="primary")
                        delete_model_btn = gr.Button("Delete Selected Model", variant="stop")
                
                model_info = gr.Textbox(
                    label="Model Information",
                    interactive=False,
                    lines=6
                )
                
                gr.Markdown("### Available Models")
                models_table = gr.Dataframe(
                    headers=["Name", "Size", "Modified At", "Digest"],
                    interactive=False
                )
                
                refresh_models_btn = gr.Button("🔄 Refresh Models List")
                
                # Event handlers
                refresh_models.click(
                    fn=update_model_selector,
                    outputs=[model_selector]
                )
                
                refresh_models_btn.click(
                    fn=update_models_table,
                    outputs=[models_table]
                )
                
                model_selector.change(
                    fn=get_model_info,
                    inputs=[model_selector],
                    outputs=[model_info]
                )
                
                pull_model_btn.click(
                    fn=pull_model,
                    inputs=[new_model_name],
                    outputs=[model_info]
                ).then(
                    fn=lambda: (update_model_selector(), update_models_table()),
                    outputs=[model_selector, models_table]
                )
                
                delete_model_btn.click(
                    fn=delete_model,
                    inputs=[model_selector],
                    outputs=[model_info]
                ).then(
                    fn=lambda: (update_model_selector(), update_models_table()),
                    outputs=[model_selector, models_table]
                )
        
                # Initialize models table
                models_table.value = update_models_table()
        
        # Chat handlers
        submit.click(
            fn=chat_interface,
            inputs=[msg, chatbot, system_prompt, temperature, model_selector],
            outputs=[chatbot, chatbot, msg]
        )
        
        msg.submit(
            fn=chat_interface,
            inputs=[msg, chatbot, system_prompt, temperature, model_selector],
            outputs=[chatbot, chatbot, msg]
        )
        
        clear.click(lambda: ([], [], ""), outputs=[chatbot, chatbot, msg])
    
    return iface

# Launch the app
if __name__ == "__main__":
    iface = create_gradio_interface()
    iface.queue()  # Enable queuing for better handling of concurrent requests
    iface.launch(
        height=600,
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        root_path="",
        ssl_verify=False,
        allowed_paths=[],
        quiet=True
    )