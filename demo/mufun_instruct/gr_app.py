import os
import gradio as gr
import tempfile
import requests
import time
import torch
import gc
import threading
from urllib.parse import urlparse
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

# --- ModelManager Class (from the second file) ---
class ModelManager:
    """
    Manages the loading and unloading of the model to conserve VRAM.
    The model is loaded on the first request and unloaded after a period of inactivity.
    """
    def __init__(self, hf_path, unload_timeout=300):
        """
        Initializes the model manager.
        :param hf_path: The Hugging Face model path.
        :param unload_timeout: The timeout in seconds to unload the model after no activity.
        """
        self.hf_path = hf_path
        self.unload_timeout = unload_timeout
        self.model = None
        self.tokenizer = None
        self.model_lock = threading.Lock()  # Thread lock to prevent concurrent loading/unloading conflicts
        self.unload_timer = None
        print("ModelManager initialized. Model will be loaded on the first request.")

    def _load_model(self):
        """Internal method: loads the model and tokenizer into VRAM."""
        if self.model is None:
            print("Loading model into VRAM...")
            start_time = time.time()
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.hf_path, use_fast=False, trust_remote_code=True)
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    llm_int8_skip_modules=["lm_head", 'vision_tower', 'connector']
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.hf_path,
                    trust_remote_code=True,
                    torch_dtype="bfloat16",
                    device_map="auto",
                    quantization_config=quantization_config
                )
                end_time = time.time()
                print(f"Model loaded successfully in {end_time - start_time:.2f} seconds.")
                print(f"Memory footprint: {self.model.get_memory_footprint()} bytes")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
                self.tokenizer = None

    def _unload_model(self):
        """Internal method: unloads the model from VRAM."""
        with self.model_lock:
            if self.model is not None:
                print("Unloading model from VRAM...")
                start_time = time.time()
                del self.model
                del self.tokenizer
                self.model = None
                self.tokenizer = None
                gc.collect()
                torch.cuda.empty_cache()
                end_time = time.time()
                print(f"Model unloaded successfully in {end_time - start_time:.2f} seconds.")
            self.unload_timer = None

    def get_model(self):
        """
        Gets the model and tokenizer. This is the main interface for external calls.
        It handles the model loading and unloading schedule.
        """
        with self.model_lock:
            if self.unload_timer:
                self.unload_timer.cancel()
                print("New request received. Canceled a pending model unload.")
            if self.model is None:
                self._load_model()
            print(f"Scheduling model unload in {self.unload_timeout} seconds.")
            self.unload_timer = threading.Timer(self.unload_timeout, self._unload_model)
            self.unload_timer.start()
            return self.model, self.tokenizer

# --- Model and App Configuration ---
# NOTE: Replace with your actual Hugging Face model path
hf_path = 'Yi3852/MuFun-Instruct'
model_manager = ModelManager(hf_path, unload_timeout=300)

# --- Main Processing Function ---
def process_audio(audio_input, url_or_path, use_segment, start_time, end_time, prompt, temperature, max_tokens):
    """
    Processes the audio input (file, URL, or path) and generates a response from the model.
    """
    audio_file = None
    temp_file_to_clean = None

    if not audio_input and not url_or_path:
        return "Error: Please upload an audio file or provide a URL/path."

    try:
        # Determine the audio source
        if audio_input is not None:
            audio_file = audio_input
        elif url_or_path:
            parsed = urlparse(url_or_path)
            if parsed.scheme in ('http', 'https'):
                # Download URL to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_f:
                    print(f"Downloading audio from URL: {url_or_path}")
                    response = requests.get(url_or_path)
                    response.raise_for_status() # Raise an exception for bad status codes
                    temp_f.write(response.content)
                    audio_file = temp_f.name
                    temp_file_to_clean = audio_file
                    print(f"Audio downloaded to temporary file: {audio_file}")
            else:
                # Use local path
                audio_file = url_or_path
        
        if not audio_file or not os.path.exists(audio_file):
             return f"Error: Audio file not found at '{audio_file}'"

        # Handle segmentation
        segs = None
        if use_segment and start_time is not None and end_time is not None:
            segs = [float(start_time), float(end_time)]

        # Get model and tokenizer from the manager
        print("Requesting model from manager...")
        model, tokenizer = model_manager.get_model()

        if model is None or tokenizer is None:
            return "Error: Model could not be loaded. Please check the server logs for details."
        
        print("Model and tokenizer acquired. Starting generation...")
        
        # Call the model's chat function with the specified interface
        response = model.chat(
            tokenizer=tokenizer,
            prompt=prompt,
            audio_files=[audio_file], # Pass as a list as per the 'audio_files' name
            segs=segs,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        
        print("--- Interaction Details ---")
        print(f"Audio file: {audio_file}")
        print(f"Prompt: {prompt}")
        print(f"Segments: {segs}")
        print(f"Response: {response}")
        print("---------------------------")
        
        return response

    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {str(e)}"
    finally:
        # Clean up the temporary file if one was created
        if temp_file_to_clean and os.path.exists(temp_file_to_clean):
            os.remove(temp_file_to_clean)
            print(f"Cleaned up temporary file: {temp_file_to_clean}")

# --- Example Data (with English prompts) ---
examples = [
    [None, "examples/Octopath Traveler.flac", False, None, None, "What instruments are used in this piece of music?", 0.5, 512],
    [None, "examples/Never Gonna Give You Up.mp3", True, 15.0, 45.0, "Can you listen to this music clip and tell me its lyrics?", 0.5, 512],
    [None, "examples/Yesterday Once More.mp3", False, None, None, "What is the emotional tone of this song?", 0.5, 512],
    [None, "examples/Nothing's Gonna Change My Love for You.mp3", False, None, None, "How is the rhythm of this music?", 0.5, 512],
    # [None, "examples/108147.0.mp3", True, 253.05, 275.89, "Can you listen to this music clip and tell me its lyrics?", 0.5, 512],
    # [None, "examples/seed.mp3", False, None, None, "What instruments are used in this piece of music?", 0.5, 512],
    # [None, "examples/1410707394.0.mp3", False, None, None, "What is the emotional tone of this song?", 0.5, 512],
    # [None, "examples/524003578.0.mp3", False, None, None, "How is the rhythm of this music?", 0.5, 512],
]

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    # [MuFun-Instruct](https://huggingface.co/Yi3852/MuFun-Instruct) Demo
    You can upload a file or provide a URL. (currently supported audio formats: '.wav', '.mp3', '.flac', '.opus', '.ogg')
    **Note:** The model is loaded on the first request and unloaded after 5 minutes of inactivity to save VRAM.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input Area
            audio_input = gr.Audio(type="filepath", label="Upload Audio File")
            url_or_path = gr.Textbox(label="Or enter Audio URL/Path")
            
            use_segment = gr.Checkbox(label="Select Audio Segment")
            
            with gr.Row(visible=False) as segment_row:
                start_time = gr.Number(label="Start Time (s)")
                end_time = gr.Number(label="End Time (s)")
            
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your question...")
            
            with gr.Accordion("Advanced Options", open=False):
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.1, label="Temperature")
                max_tokens = gr.Slider(minimum=64, maximum=2048, value=512, step=64, label="Max New Tokens")
            
            with gr.Row():
                submit_btn = gr.Button("Generate Answer", variant="primary")
                clear_btn = gr.Button("Clear / Stop")
        
        with gr.Column(scale=1):
            # Output Area
            output = gr.Textbox(label="Model Output", lines=20, interactive=False)
    
    # Examples Area
    gr.Examples(
        examples=examples,
        inputs=[audio_input, url_or_path, use_segment, start_time, end_time, prompt, temperature, max_tokens],
        outputs=output,
        fn=process_audio,
        cache_examples=False
    )
    
    # --- UI Logic ---
    def update_segment_visibility(use_seg):
        return gr.update(visible=use_seg)
    
    use_segment.change(
        fn=update_segment_visibility,
        inputs=use_segment,
        outputs=segment_row
    )
    
    submit_event = submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, url_or_path, use_segment, start_time, end_time, prompt, temperature, max_tokens],
        outputs=output
    )
    
    def clear_inputs():
        # Corresponds to outputs: [audio_input, url_or_path, use_segment, start_time, end_time, prompt, temperature, max_tokens, output]
        return [None, "", False, None, None, "", 0.5, 512, ""]
    
    clear_btn.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[audio_input, url_or_path, use_segment, start_time, end_time, prompt, temperature, max_tokens, output],
        cancels=[submit_event] # Also cancel any ongoing generation
    )

# --- Launch the App ---
if __name__ == "__main__":
    # Ensure the examples directory exists for the app to function correctly
    if not os.path.exists("examples"):
        print("Warning: 'examples' directory not found. The examples might not work unless audio files are available.")
    
    demo.launch(share=True, allowed_paths=["."], server_port=7861)