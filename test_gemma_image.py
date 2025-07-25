import torch
from PIL import Image
import requests
from requests.exceptions import RequestException
import time
from utils import load_model_and_processor

model, processor, device = load_model_and_processor()

# Sample image URL (can be replaced with your own)
image_url = "https://images.unsplash.com/photo-1506744038136-46273834b3fb"

try:
    response = requests.get(image_url, stream=True)
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    image = Image.open(response.raw)
except RequestException as e:
    print(f"Error fetching image: {e}")
    exit()


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
start_time = time.time()
with torch.no_grad():
    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:] # remove input prompt from output

# Decode output
decoded = processor.decode(generation, skip_special_tokens=True)
elapsed = time.time() - start_time
try:
    print("Model output:", decoded)
    print(f"Inference time: {elapsed:.2f} seconds")
finally:
    image.close()