import gradio as gr
import json
import time
import numpy as np
import os
import torch
import gc
import threading

from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

# We create a class to handle model loading, unloading, and state management

class ModelManager:
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
        print("ModelManager initialized. Model will be loaded on first request.")

    def _load_model(self):
        """Internal method: loads the model and tokenizer into VRAM"""
        if self.model is None:
            print("Loading model into VRAM...")
            start_time = time.time()
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.hf_path, use_fast=False)
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
        """Internal method: unloads the model from VRAM"""
        # Acquire a lock to ensure no other threads are accessing the model during unload
        with self.model_lock:
            if self.model is not None:
                print("Unloading model from VRAM...")
                start_time = time.time()
                del self.model
                del self.tokenizer
                self.model = None
                self.tokenizer = None
                gc.collect()  # Garbage collection
                torch.cuda.empty_cache()  # Clear PyTorch's CUDA cache
                end_time = time.time()
                print(f"Model unloaded successfully in {end_time - start_time:.2f} seconds.")
            # Ensure the timer is cleared
            self.unload_timer = None

    def get_model(self):
        """
        Gets the model and tokenizer. This is the main interface for external calls.
        It handles the model loading and unloading schedule.
        """
        with self.model_lock:
            # If there is a pending unload task, cancel it because we have received a new request
            if self.unload_timer:
                self.unload_timer.cancel()
                print("New request received. Canceled a pending model unload.")

            # If the model is not loaded, load it
            if self.model is None:
                self._load_model()

            # (Re)set the unload timer
            print(f"Scheduling model unload in {self.unload_timeout} seconds.")
            self.unload_timer = threading.Timer(self.unload_timeout, self._unload_model)
            self.unload_timer.start()

            return self.model, self.tokenizer

hf_path = 'Yi3852/MuFun-ACEStep'
# Set the timeout to 300 seconds (5 minutes)
model_manager = ModelManager(hf_path, unload_timeout=300)

inp = '<audio>\nDeconstruct this song, listing its tags and lyrics. Directly output a JSON object with prompt and lyrics fields, without any additional explanations or text.'

EXAMPLE_DIR = "examples"
if not os.path.exists(EXAMPLE_DIR):
    os.makedirs(EXAMPLE_DIR)
example_path_1 = os.path.join(EXAMPLE_DIR, "ace_output.wav")
example_path_2 = os.path.join(EXAMPLE_DIR, "yesterday.flac")
example_path_3 = os.path.join(EXAMPLE_DIR, "梦中人.mp3")


def generate_music_data(audio_filepath):
    if not audio_filepath:
        return "Please upload an audio file first.", ""

    print(f"Received request for audio file: {audio_filepath}")
    
    # Get the model from the manager; this step automatically handles loading and scheduled unloading
    model, tokenizer = model_manager.get_model()
    
    # Check if the model was loaded successfully
    if model is None or tokenizer is None:
        error_msg = "Model could not be loaded. Please check the logs."
        return error_msg, error_msg
        
    try:
        # Use the obtained model for inference
        res = model.chat(prompt=inp, audio_files=audio_filepath, segs=None, tokenizer=tokenizer, temperature=0.2)
        print(f"Model response: {res}")
        json_output = json.loads(res)
    except Exception as e:
        print(f"Error during model inference: {e}")
        return "Error during model inference", str(e)
    
    prompt = json_output.get("prompt", "N/A")
    lyrics = json_output.get("lyrics", "N/A")
    
    return prompt, lyrics

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎵 Music to Prompt & Lyrics Generator
        Upload a song or select an example below. The model will analyze the audio 
        and generate a descriptive prompt and the corresponding lyrics.
        **Note:** The model will be automatically loaded on the first request and unloaded after 5 minutes of inactivity to save VRAM.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                type="filepath", 
                label="Upload Your Song",
                sources=["upload", "microphone"]
            )
            gr.Examples(
                examples=[example_path_1, example_path_2, example_path_3],
                inputs=audio_input,
                label="Example Songs"
            )
            submit_btn = gr.Button("✨ Generate", variant="primary")

        with gr.Column(scale=2):
            prompt_output = gr.Textbox(
                label="Generated Prompt",
                info="A set of descriptive tags for the music."
            )
            lyrics_output = gr.Textbox(
                label="Generated Lyrics", 
                lines=15,
                info="The lyrics for the music."
            )

    submit_btn.click(
        fn=generate_music_data,
        inputs=audio_input,
        outputs=[prompt_output, lyrics_output]
    )

if __name__ == "__main__":
    demo.launch(share=True, allowed_paths=["."])