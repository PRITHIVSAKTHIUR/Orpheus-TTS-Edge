# **Orpheus-Edge-TTS-Demo**

https://github.com/user-attachments/assets/1896b9bc-dccf-4180-9e1d-5e658cf4e3e5

Orpheus TTS, a Llama-based Speech-LLM designed for high-quality, empathetic text-to-speech generation. This model has been fine-tuned to deliver human-level speech synthesis 

> [!WARNING]
> Don't forget to add the `HF_TOKEN` to the environment to access the gated Hugging Face models.

## Features

### 1. **Multimodal Input Support**
   - **Text Input**: Process text-based queries with DeepHermes Llama for natural language understanding.
   - **Image Input**: Analyze and describe images using Qwen2-VL.
   - **Video Input**: Process videos by extracting key frames and summarizing content.

### 2. **Advanced Text-to-Speech (TTS)**
   - **Orpheus TTS**: Generate realistic speech with customizable voices (`tara`, `dan`, `emma`, `josh`).
   - **Emotion Support**: Add emotions like `<laugh>`, `<sigh>`, `<gasp>`, etc., to make the speech more expressive.
   - **Direct TTS**: Convert text to speech directly using `@<voice>-tts` (e.g., `@tara-tts`).
   - **LLM-Augmented TTS**: Generate a response using DeepHermes Llama and then convert it to speech using `@<voice>-llm` (e.g., `@tara-llm`).

### 3. **Video Processing**
   - Use the `@video-infer` command to analyze and summarize video content. The system extracts key frames and processes them with Qwen2-VL.

### 4. **Customizable Parameters**
   - Adjust generation parameters like `temperature`, `top-p`, `top-k`, and `repetition penalty` to fine-tune responses.

---

## Usage

### Commands
1. **Direct TTS**:
   - Use `@<voice>-tts` to directly convert text to speech.
   - Example: `@tara-tts Hey, I’m Tara, [laugh] and I’m a speech generation model!`

2. **LLM-Augmented TTS**:
   - Use `@<voice>-llm` to generate a response with DeepHermes Llama and then convert it to speech.
   - Example: `@tara-llm Explain the causes of rainbows.`

3. **Video Processing**:
   - Use `@video-infer` to analyze and summarize video content.
   - Example: `@video-infer Summarize the event in this video.`

4. **Regular Chat**:
   - Input text or upload images/videos for multimodal processing.
   - Example: `Write a Python program for array rotation.`

---

## Examples

### Text-to-Speech (TTS)
- `@josh-tts Hey! I’m Josh, [gasp] and wow, did I just surprise you with my realistic voice?`
- `@emma-tts Hey, I’m Emma, [sigh] and yes, I can talk just like a person… even when I’m tired.`

### LLM-Augmented TTS
- `@dan-llm Explain the General Relativity theorem in short.`
- `@tara-llm Who is Nikola Tesla, and why did he die?`

### Video Processing
- `@video-infer Summarize the event in this video.`
- `@video-infer Describe the video.`

### Multimodal Input
- `summarize the letter` (with an uploaded image).
- `Explain the causes of rainbows` (with an uploaded video).

---

## Setup

1. **Install Dependencies**:
   Ensure you have the required Python packages installed:
   ```bash
   pip install torch gradio transformers huggingface-hub snac dotenv
   ```

2. **Environment Variables**:
   - Set `MAX_INPUT_TOKEN_LENGTH` in `.env` to control the maximum input token length for the LLM.

3. **Run the Application**:
   ```bash
   python app.py
   ```

4. **Access the Interface**:
   - The Gradio interface will launch locally. Use the provided examples or input your own queries.

---

## Models Used

1. **DeepHermes Llama**:
   - A fine-tuned Llama model for natural language understanding and generation.
   - Model ID: `prithivMLmods/DeepHermes-3-Llama-3-3B-Preview-abliterated`.

2. **Qwen2-VL**:
   - A multimodal model for image and video processing.
   - Model ID: `prithivMLmods/Qwen2-VL-OCR2-2B-Instruct`.

3. **Orpheus TTS**:
   - A high-quality text-to-speech model for generating realistic speech.
   - Model ID: `canopylabs/orpheus-3b-0.1-ft`.

4. **SNAC**:
   - A neural audio codec used for decoding TTS outputs.
   - Model ID: `hubertsiuzdak/snac_24khz`.

---

## Customization

- **Voices**: Choose from `tara`, `dan`, `emma`, or `josh` for TTS.
- **Emotions**: Add emotions like `<laugh>`, `<sigh>`, `<gasp>`, etc., to make the speech more expressive.
- **Generation Parameters**: Adjust `temperature`, `top-p`, `top-k`, and `repetition penalty` to fine-tune responses.

---

## Notes

- **Hardware Requirements**: A GPU is recommended for optimal performance, especially for TTS and video processing.
- **Limitations**:
  - Video processing is limited to 10 key frames per video.
  - TTS generation may take longer for longer texts.

---

## Acknowledgments

- **Hugging Face** for providing the models and tools.
- **Gradio** for the intuitive interface.
- **SNAC** and **Orpheus TTS** for high-quality speech synthesis.
