import os
import random
import uuid
import json
import time
import asyncio
from threading import Thread

import gradio as gr
import spaces
import torch
import numpy as np
from PIL import Image
import cv2

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
from transformers.image_utils import load_image

# Additional imports for new TTS
from snac import SNAC
from huggingface_hub import snapshot_download
from dotenv import load_dotenv
load_dotenv()

# Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tts_device = "cuda" if torch.cuda.is_available() else "cpu"  # for SNAC and Orpheus TTS

# Load DeepHermes Llama (chat/LLM) model
hermes_model_id = "prithivMLmods/DeepHermes-3-Llama-3-3B-Preview-abliterated"
hermes_llm_tokenizer = AutoTokenizer.from_pretrained(hermes_model_id)
hermes_llm_model = AutoModelForCausalLM.from_pretrained(
    hermes_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
hermes_llm_model.eval()

# Load Qwen2-VL processor and model for multimodal tasks (e.g. video processing)
MODEL_ID_QWEN = "prithivMLmods/Qwen2-VL-OCR2-2B-Instruct" 
processor = AutoProcessor.from_pretrained(MODEL_ID_QWEN, trust_remote_code=True)
model_m = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID_QWEN,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to("cuda").eval()

# Load Orpheus TTS model and SNAC for TTS synthesis
print("Loading SNAC model...")
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
snac_model = snac_model.to(tts_device)

tts_model_name = "canopylabs/orpheus-3b-0.1-ft"
# Download only model config and safetensors
snapshot_download(
    repo_id=tts_model_name,
    allow_patterns=[
        "config.json",
        "*.safetensors",
        "model.safetensors.index.json",
    ],
    ignore_patterns=[
        "optimizer.pt",
        "pytorch_model.bin",
        "training_args.bin",
        "scheduler.pt",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.*"
    ]
)
orpheus_tts_model = AutoModelForCausalLM.from_pretrained(tts_model_name, torch_dtype=torch.bfloat16)
orpheus_tts_model.to(tts_device)
orpheus_tts_tokenizer = AutoTokenizer.from_pretrained(tts_model_name)
print(f"Orpheus TTS model loaded to {tts_device}")

# Some global parameters for chat responses
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

# (Image generation related code has been fully removed.)

MAX_SEED = np.iinfo(np.int32).max

# Utility functions
def save_image(img: Image.Image) -> str:
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def progress_bar_html(label: str) -> str:
    return f'''
<div style="display: flex; align-items: center;">
    <span style="margin-right: 10px; font-size: 14px;">{label}</span>
    <div style="width: 110px; height: 5px; background-color: #FFA07A; border-radius: 2px; overflow: hidden;">
        <div style="width: 100%; height: 100%; background-color: #FF4500; animation: loading 1.5s linear infinite;"></div>
    </div>
</div>
<style>
@keyframes loading {{
    0% {{ transform: translateX(-100%); }}
    100% {{ transform: translateX(100%); }}
}}
</style>
    '''

def downsample_video(video_path):
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
    for i in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            timestamp = round(i / fps, 2)
            frames.append((pil_image, timestamp))
    vidcap.release()
    return frames

def clean_chat_history(chat_history):
    cleaned = []
    for msg in chat_history:
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            cleaned.append(msg)
    return cleaned

# New TTS functions (SNAC/Orpheus pipeline)
def process_prompt(prompt, voice, tokenizer, device):
    prompt = f"{voice}: {prompt}"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    start_token = torch.tensor([[128259]], dtype=torch.int64)  # Start of human
    end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)  # End markers
    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
    attention_mask = torch.ones_like(modified_input_ids)
    return modified_input_ids.to(device), attention_mask.to(device)

def parse_output(generated_ids):
    token_to_find = 128257
    token_to_remove = 128258
    token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)
    if len(token_indices[1]) > 0:
        last_occurrence_idx = token_indices[1][-1].item()
        cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
    else:
        cropped_tensor = generated_ids
    processed_rows = []
    for row in cropped_tensor:
        masked_row = row[row != token_to_remove]
        processed_rows.append(masked_row)
    code_lists = []
    for row in processed_rows:
        row_length = row.size(0)
        new_length = (row_length // 7) * 7
        trimmed_row = row[:new_length]
        trimmed_row = [t - 128266 for t in trimmed_row]
        code_lists.append(trimmed_row)
    return code_lists[0]

def redistribute_codes(code_list, snac_model):
    device = next(snac_model.parameters()).device
    layer_1 = []
    layer_2 = []
    layer_3 = []
    for i in range((len(code_list)+1)//7):
        layer_1.append(code_list[7*i])
        layer_2.append(code_list[7*i+1]-4096)
        layer_3.append(code_list[7*i+2]-(2*4096))
        layer_3.append(code_list[7*i+3]-(3*4096))
        layer_2.append(code_list[7*i+4]-(4*4096))
        layer_3.append(code_list[7*i+5]-(5*4096))
        layer_3.append(code_list[7*i+6]-(6*4096))
    codes = [
        torch.tensor(layer_1, device=device).unsqueeze(0),
        torch.tensor(layer_2, device=device).unsqueeze(0),
        torch.tensor(layer_3, device=device).unsqueeze(0)
    ]
    audio_hat = snac_model.decode(codes)
    return audio_hat.detach().squeeze().cpu().numpy()

def generate_speech(text, voice, temperature, top_p, repetition_penalty, max_new_tokens):
    if not text.strip():
        return None
    try:
        # Removed in-function progress calls to maintain UI consistency.
        input_ids, attention_mask = process_prompt(text, voice, orpheus_tts_tokenizer, tts_device)
        with torch.no_grad():
            generated_ids = orpheus_tts_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=1,
                eos_token_id=128258,
            )
        code_list = parse_output(generated_ids)
        audio_samples = redistribute_codes(code_list, snac_model)
        return (24000, audio_samples)
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None

# Main generate function for the chat interface
@spaces.GPU
def generate(
    input_dict: dict,
    chat_history: list[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
):
    """
    Generates chatbot responses with support for multimodal input, video processing,
    TTS, and LLM-augmented TTS.
    
    Trigger commands:
      - "@video-infer": process video.
      - "@<voice>-tts": directly convert text to speech.
      - "@<voice>-llm": infer with the DeepHermes Llama model then convert to speech.
    """
    text = input_dict["text"]
    files = input_dict.get("files", [])
    lower_text = text.strip().lower()

    # Branch for video processing.
    if lower_text.startswith("@video-infer"):
        prompt = text[len("@video-infer"):].strip()
        if files:
            video_path = files[0]
            frames = downsample_video(video_path)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
            for frame in frames:
                image, timestamp = frame
                image_path = f"video_frame_{uuid.uuid4().hex}.png"
                image.save(image_path)
                messages[1]["content"].append({"type": "text", "text": f"Frame {timestamp}:"})
                messages[1]["content"].append({"type": "image", "url": image_path})
        else:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to("cuda")
        streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }
        thread = Thread(target=model_m.generate, kwargs=generation_kwargs)
        thread.start()
        buffer = ""
        yield progress_bar_html("Processing video with Qwen2VL")
        for new_text in streamer:
            buffer += new_text.replace("<|im_end|>", "")
            time.sleep(0.01)
            yield buffer
        return

    # Define TTS and LLM tag mappings.
    tts_tags = {"@tara-tts": "tara", "@dan-tts": "dan", "@josh-tts": "josh", "@emma-tts": "emma"}
    llm_tags = {"@tara-llm": "tara", "@dan-llm": "dan", "@josh-llm": "josh", "@emma-llm": "emma"}

    # Branch for direct TTS (no LLM inference).
    for tag, voice in tts_tags.items():
        if lower_text.startswith(tag):
            text = text[len(tag):].strip()
            yield progress_bar_html("Processing with Orpheus")
            audio_output = generate_speech(text, voice, temperature, top_p, repetition_penalty, max_new_tokens)
            yield gr.Audio(audio_output, autoplay=True)
            return

    # Branch for LLM-augmented TTS.
    for tag, voice in llm_tags.items():
        if lower_text.startswith(tag):
            text = text[len(tag):].strip()
            conversation = [{"role": "user", "content": text}]
            input_ids = hermes_llm_tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt")
            if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
                input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
            input_ids = input_ids.to(hermes_llm_model.device)
            streamer = TextIteratorStreamer(hermes_llm_tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = {
                "input_ids": input_ids,
                "streamer": streamer,
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "top_p": top_p,
                "top_k": 50,
                "temperature": temperature,
                "num_beams": 1,
                "repetition_penalty": repetition_penalty,
            }
            t = Thread(target=hermes_llm_model.generate, kwargs=generation_kwargs)
            t.start()
            outputs = []
            for new_text in streamer:
                outputs.append(new_text)
            final_response = "".join(outputs)
            yield progress_bar_html("Processing with Orpheus")
            audio_output = generate_speech(final_response, voice, temperature, top_p, repetition_penalty, max_new_tokens)
            yield gr.Audio(audio_output, autoplay=True)
            return

    # Default branch for regular chat (text and multimodal without TTS).
    conversation = clean_chat_history(chat_history)
    conversation.append({"role": "user", "content": text})
    # If files are provided, only non-image files (e.g. video) are processed via Qwen2VL.
    if files:
        # Process files using the processor (this branch no longer handles image generation)
        if len(files) > 1:
            inputs_list = [load_image(image) for image in files]
        elif len(files) == 1:
            inputs_list = [load_image(files[0])]
        else:
            inputs_list = []
        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in inputs_list],
                {"type": "text", "text": text},
            ]
        }]
        prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt_full], images=inputs_list, return_tensors="pt", padding=True).to("cuda")
        streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {**inputs, "streamer": streamer, "max_new_tokens": max_new_tokens}
        thread = Thread(target=model_m.generate, kwargs=generation_kwargs)
        thread.start()
        buffer = ""
        yield progress_bar_html("Processing with Qwen2VL")
        for new_text in streamer:
            buffer += new_text.replace("<|im_end|>", "")
            time.sleep(0.01)
            yield buffer
    else:
        input_ids = hermes_llm_tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt")
        if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
            input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
            gr.Warning(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
        input_ids = input_ids.to(hermes_llm_model.device)
        streamer = TextIteratorStreamer(hermes_llm_tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            "input_ids": input_ids,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "num_beams": 1,
            "repetition_penalty": repetition_penalty,
        }
        t = Thread(target=hermes_llm_model.generate, kwargs=generation_kwargs)
        t.start()
        outputs = []
        yield progress_bar_html("Processing with DeepHermes LLM")
        for new_text in streamer:
            outputs.append(new_text)
            yield "".join(outputs)
        final_response = "".join(outputs)
        yield final_response

# Gradio Interface
demo = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS),
        gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.6),
        gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9),
        gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50),
        gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.2),
    ],
    examples=[
        ["@josh-tts Hey! I’m Josh, [gasp] and wow, did I just surprise you with my realistic voice?"],
        ["@dan-llm Explain the General Relativity theorem in short"],
        ["@emma-tts Hey, I’m Emma, [sigh] and yes, I can talk just like a person… even when I’m tired."],
        ["@josh-llm What causes rainbows to form?"],
        ["@dan-tts Yo, I’m Dan, [groan] and yes, I can even sound annoyed if I have to."],
        ["Write python program for array rotation"],
        [{"text": "summarize the letter", "files": ["examples/1.png"]}],
        ["@tara-tts Hey there, my name is Tara, [laugh] and I’m a speech generation model that can sound just like you!"],
        ["@tara-llm Who is Nikola Tesla, and why did he die?"],
        ["@emma-llm Explain the causes of rainbows"],
        [{"text": "@video-infer Summarize the event in video", "files": ["examples/sky.mp4"]}],
        [{"text": "@video-infer Describe the video", "files": ["examples/Missing.mp4"]}],
    ],
    cache_examples=False,
    type="messages",
    description="# **Orpheus Edge🧤** `voice: tara, dan, emma, josh` \n `emotion: <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>. Use @video-infer, orpheus: @<voice>-tts, or @<voice>-llm triggers llm response`",
    fill_height=True,
    textbox=gr.MultimodalTextbox(label="Query Input", file_types=["image", "video"], file_count="multiple", placeholder="‎ Use @tara-tts/@dan-tts for direct TTS or @tara-llm/@dan-llm for LLM+TTS, etc."),
    stop_btn="Stop Generation",
    multimodal=True,
)

if __name__ == "__main__":
    demo.queue(max_size=20).launch(share=True)
