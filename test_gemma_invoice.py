import torch
from transformers import AutoProcessor, AutoModelForSeq2SeqLM  # Use AutoModelForSeq2SeqLM for generic seq2seq tasks
from PIL import Image
import time
import os
import concurrent.futures
from torch.quantization import quantize_dynamic  # Add quantization import

# Set the number of threads and ensure it uses available CPU cores efficiently
os.environ["OMP_NUM_THREADS"] = "8"  # Experiment with values like 4, 8, or 16
torch.set_num_threads(8)

device = torch.device("cpu")  # Explicitly use CPU
model_id = "google/gemma-3n-e2b-it"  # Official model

# Load the model with dynamic quantization (reduce model size and speed up inference)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id
)
model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)  # Apply quantization to linear layers
model = model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# List of image paths to process in parallel
image_paths = [
    "input_images/invoice13.jpg",
    "input_images/invoice14.jpg",
    "input_images/invoice15.jpg"
]

def run_inference(image_path):
    try:
        image = Image.open(image_path).convert("RGB").resize((768, 768))
        
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Extract invoice_number, date, vendor, total_amount, and line_items (description, quantity, unit_price, line_total) as JSON."}
                ]
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
        
        start_time = time.time()
        with torch.no_grad():
            generation = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                use_cache=True  # Enable caching for faster inference
            )
        
        generation = generation[0][inputs["input_ids"].shape[-1]:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        elapsed = time.time() - start_time
        return image_path, decoded, elapsed
    except Exception as e:
        # Handle exception if something goes wrong
        print(f"Error processing {image_path}: {e}")
        return image_path, str(e), 0  # Return error info

# Run inference in parallel for all images
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_inference, image_paths))

# Print results
for image_path, decoded, elapsed in results:
    print(f"Image: {image_path}")
    print("Model output:", decoded)
    print(f"Inference time: {elapsed:.2f} seconds\n")
