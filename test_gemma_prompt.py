import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration

model_id = "google/gemma-3n-e2b-it"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
model = Gemma3nForConditionalGeneration.from_pretrained(model_id).to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Prepare prompt (user query)
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "who is our current PM?"}]  # Prompt to test model.
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
