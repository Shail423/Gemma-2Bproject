import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
import time

model_id = "google/gemma-3n-e2b-it"
device = "cpu"
print("Using device:", device)   
# Load model and processor
model = Gemma3nForConditionalGeneration.from_pretrained(model_id).to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Load your invoice image (replace 'invoice.jpg' with your file path)
image_path = "input_images/invoice13.jpg"  # Updated with user-provided invoice image path
image = Image.open(image_path)

# Resize the image to 768x768 for potentially higher accuracy
image = image.resize((768, 768))

# Prepare prompt for structured JSON extraction
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert at extracting structured data from documents. Return the result as a JSON object with fields: invoice_number, date, vendor, total_amount, and line_items (list of {description, quantity, unit_price, line_total})."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Extract all invoice data and return as JSON."}
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

# Run inference and time it
start_time = time.time()
with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    generation = generation[0][input_len:]
# Decode output
decoded = processor.decode(generation, skip_special_tokens=True)
elapsed = time.time() - start_time
print("Model output:", decoded)
print(f"Inference time: {elapsed:.2f} seconds") 