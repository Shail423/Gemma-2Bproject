import torch
import time
from utils import load_model_and_processor

# Load model, processor, and device
model, processor, device = load_model_and_processor()

# Prepare prompt (user query)
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "who is our current Prime Minister of India?"}]  # Prompt to test model.
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
    generation = generation[0][input_len:]

# Decode output
decoded = processor.decode(generation, skip_special_tokens=True)
elapsed = time.time() - start_time
print("Model output:", decoded)
print(f"Inference time: {elapsed:.2f} seconds")
