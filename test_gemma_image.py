import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
import requests

model_id = "google/gemma-3n-e2b-it"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
model = Gemma3nForConditionalGeneration.from_pretrained(model_id).to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Sample image URL (can be replaced with your own)
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

# Prepare prompt
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

# Prepare input for the model
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(device)

input_len = inputs["input_ids"].shape[-1]

# Run inference
with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]

# Decode output
decoded = processor.decode(generation, skip_special_tokens=True)
print("Model output:", decoded) 