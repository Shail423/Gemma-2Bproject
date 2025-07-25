from utils import load_model_and_processor
import torch
import time

model, processor, device = load_model_and_processor()

# Example: You describe the image in text
image_description = "A photo of a forest with tall green trees and sunlight streaming through the leaves."

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": f"Describe this image in detail: {image_description}"}]
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(device)

input_len = inputs["input_ids"].shape[-1]

start_time = time.time()
with torch.no_grad():
    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
elapsed = time.time() - start_time
print("Model output:", decoded)
print(f"Inference time: {elapsed:.2f} seconds")