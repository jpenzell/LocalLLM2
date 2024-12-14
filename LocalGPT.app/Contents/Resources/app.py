import gradio as gr
import requests
from typing import List, Dict, Optional
import logging
from pydantic import ConfigDict
import psutil
import platform
import json
import os
from datetime import datetime
import time

# Configure Pydantic
model_config = ConfigDict(arbitrary_types_allowed=True)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Chat history configuration
HISTORY_DIR = os.path.expanduser("~/.localgpt/chat_history")
os.makedirs(HISTORY_DIR, exist_ok=True)

class ChatHistory:
    def __init__(self):
        self.current_chat_id = None
        
    def save_chat(self, history: List[List[str]], model: str, system_prompt: str) -> str:
        """Save current chat history to a file"""
        if not history:
            return "No chat to save"
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chat_id = f"chat_{timestamp}"
        
        chat_data = {
            "id": chat_id,
            "timestamp": timestamp,
            "model": model,
            "system_prompt": system_prompt,
            "history": history,
            "title": history[0][0][:50] if history else "New Chat"  # Use first message as title
        }
        
        filepath = os.path.join(HISTORY_DIR, f"{chat_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)
            
        self.current_chat_id = chat_id
        return f"Chat saved: {chat_data['title']}"
        
    def load_chat(self, chat_id: str) -> Dict:
        """Load chat history from file"""
        filepath = os.path.join(HISTORY_DIR, f"{chat_id}.json")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                chat_data = json.load(f)
            self.current_chat_id = chat_id
            return chat_data
        except Exception as e:
            logging.error(f"Error loading chat {chat_id}: {e}")
            return None
            
    def list_chats(self) -> List[Dict]:
        """List all saved chats"""
        chats = []
        for filename in sorted(os.listdir(HISTORY_DIR), reverse=True):
            if filename.endswith(".json"):
                try:
                    filepath = os.path.join(HISTORY_DIR, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        chat_data = json.load(f)
                    chats.append({
                        "id": chat_data["id"],
                        "title": chat_data["title"],
                        "timestamp": chat_data["timestamp"],
                        "model": chat_data["model"]
                    })
                except Exception as e:
                    logging.error(f"Error loading chat {filename}: {e}")
        return chats
        
    def delete_chat(self, chat_id: str) -> str:
        """Delete a saved chat"""
        filepath = os.path.join(HISTORY_DIR, f"{chat_id}.json")
        try:
            os.remove(filepath)
            if self.current_chat_id == chat_id:
                self.current_chat_id = None
            return f"Chat {chat_id} deleted"
        except Exception as e:
            logging.error(f"Error deleting chat {chat_id}: {e}")
            return f"Error deleting chat: {str(e)}"

# Initialize chat history manager
chat_manager = ChatHistory()

def get_system_info() -> Dict[str, any]:
    """Get system information for model recommendations"""
    return {
        "memory_gb": psutil.virtual_memory().total / (1024 ** 3),  # RAM in GB
        "cpu_cores": psutil.cpu_count(),
        "platform": platform.system(),
        "machine": platform.machine()
    }

def get_model_requirements(model_name: str) -> Dict[str, any]:
    """Get model requirements based on model name"""
    # Default requirements for different model sizes
    requirements = {
        "tiny": {"min_ram": 4, "recommended_ram": 8},
        "small": {"min_ram": 8, "recommended_ram": 16},
        "medium": {"min_ram": 16, "recommended_ram": 32},
        "large": {"min_ram": 32, "recommended_ram": 64}
    }
    
    # Determine model size category based on name
    if any(x in model_name.lower() for x in ["tiny", "mini", "nano"]):
        return requirements["tiny"]
    elif any(x in model_name.lower() for x in ["small", "7b"]):
        return requirements["small"]
    elif any(x in model_name.lower() for x in ["medium", "13b"]):
        return requirements["medium"]
    elif any(x in model_name.lower() for x in ["large", "33b", "65b", "70b"]):
        return requirements["large"]
    else:
        return requirements["medium"]  # Default to medium requirements

def get_recommended_models() -> List[str]:
    """Get list of recommended models based on system capabilities"""
    system_info = get_system_info()
    available_ram = system_info["memory_gb"]
    
    recommended = []
    if available_ram >= 32:
        recommended.extend(["llama2:70b", "mixtral:8x7b", "llama2:13b"])
    elif available_ram >= 16:
        recommended.extend(["llama2:13b", "mistral:7b", "neural-chat:7b"])
    elif available_ram >= 8:
        recommended.extend(["mistral:7b-q4_0", "orca-mini:3b", "phi:latest"])
    else:
        recommended.extend(["orca-mini:3b-q4_0", "phi:latest"])
    
    return recommended

def fetch_available_models() -> List[Dict[str, any]]:
    """Fetch list of all available models from Ollama's library"""
    try:
        # This is a placeholder until Ollama provides an API for available models
        # For now, we'll use a curated list of common models
        models = [
            {"name": "llama2:70b", "size": "70B parameters", "type": "Large language model", "description": "Meta's largest LLaMA 2 model"},
            {"name": "mixtral:8x7b", "size": "47B parameters", "type": "Large language model", "description": "Mixtral 8x7B MoE model"},
            {"name": "llama2:13b", "size": "13B parameters", "type": "Large language model", "description": "Meta's medium LLaMA 2 model"},
            {"name": "mistral:7b", "size": "7B parameters", "type": "Large language model", "description": "Mistral AI's base model"},
            {"name": "neural-chat:7b", "size": "7B parameters", "type": "Chat model", "description": "Optimized for chat interactions"},
            {"name": "orca-mini:3b", "size": "3B parameters", "type": "Small language model", "description": "Lightweight general-purpose model"},
            {"name": "phi:latest", "size": "2.7B parameters", "type": "Small language model", "description": "Microsoft's efficient small model"}
        ]
        return models
    except Exception as e:
        logging.error(f"Error fetching available models: {e}")
        return []

def get_installed_models() -> List[Dict[str, str]]:
    """Get list of installed models from Ollama"""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        response.raise_for_status()
        data = response.json()
        if 'models' in data:
            return [{
                'name': model['name'],
                'size': model.get('size', 'N/A'),
                'modified_at': model.get('modified_at', 'N/A'),
                'digest': model.get('digest', 'N/A'),
                'status': 'installed'
            } for model in data['models']]
        return []
    except Exception as e:
        logging.error(f"Error fetching installed models: {e}")
        return []

def update_models_table():
    """Update the models table with both installed and available models"""
    installed_models = get_installed_models()
    available_models = fetch_available_models()
    recommended_models = get_recommended_models()
    
    # Create a lookup for installed models
    installed_names = {model['name'] for model in installed_models}
    
    # Combine installed and available models
    all_models = []
    
    # Add installed models
    for model in installed_models:
        model['status'] = 'Installed'
        model['recommended'] = '⭐' if model['name'] in recommended_models else ''
        all_models.append([
            model['name'],
            model['size'],
            model['status'],
            model['recommended']
        ])
    
    # Add available but not installed models
    for model in available_models:
        if model['name'] not in installed_names:
            all_models.append([
                model['name'],
                model['size'],
                'Available',
                '⭐' if model['name'] in recommended_models else ''
            ])
    
    return all_models

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

def pull_model(model_name: str, progress: gr.Progress) -> str:
    """Pull a new model from Ollama"""
    if not model_name or model_name.strip() == "":
        return "Error: No model selected"
    
    try:
        # Clean up model name
        model_name = model_name.strip()
        
        # First, check if the model exists in our available models list
        available_models = fetch_available_models()
        model_exists = any(model['name'] == model_name for model in available_models)
        
        if not model_exists:
            return f"Error: Model '{model_name}' not found in available models"
        
        # Show initial progress
        progress(0, desc=f"Starting download of {model_name}")
        logging.info(f"Starting download of model: {model_name}")
        
        # Make the pull request
        response = requests.post(
            'http://localhost:11434/api/pull',
            json={"name": model_name, "insecure": True},
            stream=True
        )
        
        if response.status_code == 400:
            error_text = response.text
            logging.error(f"Pull request failed: {error_text}")
            return f"Error pulling model: {error_text}"
        
        response.raise_for_status()
        
        # Process the streaming response
        for i, line in enumerate(response.iter_lines()):
            if line:
                try:
                    progress_data = line.decode('utf-8')
                    logging.info(f"Pull progress: {progress_data}")
                    # Update progress (using a simple counter as percentage)
                    progress((i % 100) / 100, desc=f"Downloading {model_name}...")
                except Exception as e:
                    logging.error(f"Error decoding progress: {e}")
        
        progress(1.0, desc="Download complete!")
        return f"Successfully pulled model: {model_name}"
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if "404" in error_msg:
            return f"Error: Model '{model_name}' not found"
        elif "400" in error_msg:
            return f"Error: Invalid model name or format: {model_name}"
        else:
            logging.error(f"Error pulling model {model_name}: {e}")
            return f"Error: {str(e)}"
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

def update_model_selector():
    """Update the model selector dropdown with current models"""
    installed_models = get_installed_models()
    available_models = fetch_available_models()
    
    # Combine all model names
    all_model_names = []
    installed_names = {model['name'] for model in installed_models}
    
    # Add installed models first
    all_model_names.extend(list(installed_names))
    
    # Add available models that aren't installed
    for model in available_models:
        if model['name'] not in installed_names:
            all_model_names.append(model['name'])
    
    return gr.update(choices=all_model_names, value=all_model_names[0] if all_model_names else None)

def get_installed_models_list() -> List[str]:
    """Get list of only installed models from Ollama"""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        response.raise_for_status()
        data = response.json()
        if 'models' in data:
            return [model['name'] for model in data['models']]
        return ["mistral"]  # Default fallback
    except Exception as e:
        logging.error(f"Error fetching installed models: {e}")
        return ["mistral"]  # Default fallback

def update_chat_model_selector():
    """Update the chat model selector with only installed models"""
    models = get_installed_models_list()
    return gr.update(choices=models, value=models[0] if models else None)

# Create the Gradio interface
def create_gradio_interface():
    with gr.Blocks() as iface:
        gr.Markdown("# Ollama Chat Interface")
        
        with gr.Tabs():
            with gr.TabItem("Chat"):
                # Chat interface components
                with gr.Row():
                    with gr.Column(scale=2):
                        installed_models = get_installed_models_list()
                        chat_model_selector = gr.Dropdown(
                            label="Select AI Model",
                            choices=installed_models,
                            value=installed_models[0] if installed_models else None,
                            interactive=True,
                            allow_custom_value=False
                        )
                    with gr.Column(scale=1):
                        refresh_chat_models = gr.Button("🔄 Refresh Models")
                
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
                        
                        # Chat history management
                        gr.Markdown("### Chat History")
                        with gr.Row():
                            save_chat = gr.Button("💾 Save Chat")
                            load_chat = gr.Button("📂 Load Chat")
                            delete_chat = gr.Button("🗑️ Delete Chat")
                        
                        chat_list = gr.Dropdown(
                            label="Saved Chats",
                            choices=[],
                            type="value",
                            interactive=True,
                            allow_custom_value=False
                        )
                        
                        chat_info = gr.Textbox(
                            label="Chat Info",
                            interactive=False,
                            lines=2
                        )
            
            with gr.TabItem("Model Management"):
                system_info = get_system_info()
                gr.Markdown(f"""## Model Management
                
                System Information:
                - RAM: {system_info['memory_gb']:.1f} GB
                - CPU Cores: {system_info['cpu_cores']}
                - Platform: {system_info['platform']} ({system_info['machine']})
                
                ⭐ = Recommended for your system
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        # Initialize with available models
                        initial_models = [model['name'] for model in fetch_available_models()]
                        model_selector = gr.Dropdown(
                            label="Select Model",
                            choices=initial_models,
                            value=initial_models[0] if initial_models else None,
                            interactive=True,
                            allow_custom_value=False
                        )
                        model_info = gr.Textbox(
                            label="Model Information",
                            interactive=False,
                            lines=6
                        )
                    with gr.Column(scale=1):
                        install_btn = gr.Button("Install Selected Model", variant="primary")
                        delete_btn = gr.Button("Delete Selected Model", variant="stop")
                        refresh_btn = gr.Button("🔄 Refresh Models List")
                
                gr.Markdown("""### Model Installation Notes
                - Large models (>30B parameters) may take 30+ minutes to download
                - Medium models (7B-13B) typically take 10-15 minutes
                - Small models (<7B) usually download in 5-10 minutes
                - Download times depend on your internet speed and system performance
                """)
                
                gr.Markdown("### Available Models")
                models_table = gr.Dataframe(
                    headers=["Name", "Size", "Status", "Recommended"],
                    interactive=False
                )
                
                # Event handlers
                def refresh_all():
                    table_data = update_models_table()
                    selector_update = update_model_selector()
                    return table_data, selector_update
                
                refresh_btn.click(
                    fn=refresh_all,
                    outputs=[models_table, model_selector]
                )
                
                model_selector.change(
                    fn=get_model_info,
                    inputs=[model_selector],
                    outputs=[model_info]
                )
                
                install_btn.click(
                    fn=pull_model,
                    inputs=[model_selector],
                    outputs=[model_info],
                    show_progress=True
                ).then(
                    fn=lambda: (update_models_table(), update_model_selector(), update_chat_model_selector()),
                    outputs=[models_table, model_selector, chat_model_selector]
                )
                
                delete_btn.click(
                    fn=delete_model,
                    inputs=[model_selector],
                    outputs=[model_info]
                ).then(
                    fn=lambda: (update_models_table(), update_model_selector(), update_chat_model_selector()),
                    outputs=[models_table, model_selector, chat_model_selector]
                )
        
                # Initialize models table
                models_table.value = update_models_table()
        
        # Chat handlers
        submit.click(
            fn=chat_interface,
            inputs=[msg, chatbot, system_prompt, temperature, chat_model_selector],
            outputs=[chatbot, chatbot, msg]
        )
        
        msg.submit(
            fn=chat_interface,
            inputs=[msg, chatbot, system_prompt, temperature, chat_model_selector],
            outputs=[chatbot, chatbot, msg]
        )
        
        clear.click(lambda: ([], [], ""), outputs=[chatbot, chatbot, msg])
        
        # Event handlers for chat page
        refresh_chat_models.click(
            fn=update_chat_model_selector,
            outputs=[chat_model_selector]
        )
        
        # Connect the model management refresh to also update chat model selector
        refresh_btn.click(
            fn=lambda: (update_models_table(), update_model_selector(), update_chat_model_selector()),
            outputs=[models_table, model_selector, chat_model_selector]
        )
        
        # Chat history event handlers
        def update_chat_list():
            chats = chat_manager.list_chats()
            return gr.update(
                choices=[f"{c['title']} ({c['timestamp']})" for c in chats],
                value=None
            )
        
        def save_current_chat(history, model, prompt):
            if not history:
                return "No chat to save", update_chat_list()
            result = chat_manager.save_chat(history, model, prompt)
            return result, update_chat_list()
        
        def load_selected_chat(selected):
            if not selected:
                return [], "No chat selected", None, None
            chat_id = selected.split("(")[1].strip(")").strip()
            chat_data = chat_manager.load_chat(f"chat_{chat_id}")
            if chat_data:
                return (
                    chat_data["history"],
                    f"Loaded chat from {chat_data['timestamp']}",
                    chat_data["system_prompt"],
                    chat_data["model"]
                )
            return [], "Error loading chat", None, None
        
        def delete_selected_chat(selected):
            if not selected:
                return "No chat selected", update_chat_list()
            chat_id = selected.split("(")[1].strip(")").strip()
            result = chat_manager.delete_chat(f"chat_{chat_id}")
            return result, update_chat_list()
        
        # Connect chat history event handlers
        save_chat.click(
            fn=save_current_chat,
            inputs=[chatbot, chat_model_selector, system_prompt],
            outputs=[chat_info, chat_list]
        )
        
        load_chat.click(
            fn=load_selected_chat,
            inputs=[chat_list],
            outputs=[chatbot, chat_info, system_prompt, chat_model_selector]
        )
        
        delete_chat.click(
            fn=delete_selected_chat,
            inputs=[chat_list],
            outputs=[chat_info, chat_list]
        )
        
        # Initialize chat list
        chat_list.value = update_chat_list()
    
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