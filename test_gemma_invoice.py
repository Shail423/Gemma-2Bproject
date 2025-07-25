import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import time
import os
import concurrent.futures
 
# Set the number of threads for CPU (optional, doesn't affect GPU)
os.environ["OMP_NUM_THREADS"] = "16"
torch.set_num_threads(16)
 
device = torch.device("cpu")  # Force CPU usage
model_id = "google/gemma-3n-e2b-it"
 
# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
 
# Enable PyTorch inference optimizations
torch.set_grad_enabled(False)
torch.backends.mkldnn.enabled = True
 
# List of image paths to process in parallel
image_paths = [
    "input_images/invoice13.jpg",
    # Add more image paths if needed
]
 
def run_inference(image_path):
    try:
        # Loading and processing image (Gemma does not support images, so use a text description instead)
        image_description = f"This is an invoice image located at {image_path}. Please extract invoice_number, date, vendor, total_amount, and line_items (description, quantity, unit_price, line_total) as JSON."
       
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
                    {"type": "text", "text": f"Extract all invoice data from this image description and return as JSON: {image_description}"}
                ]
            }
        ]
       
        # Prepare input for the model
        inputs = tokenizer.apply_chat_template(
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
                use_cache=True
            )
        generation = generation[0][inputs["input_ids"].shape[-1]:]
        decoded = tokenizer.decode(generation, skip_special_tokens=True)
        elapsed = time.time() - start_time
        return image_path, decoded, elapsed
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return image_path, str(e), 0
 
# Run inference in parallel for all images
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_inference, image_paths))
 
# Print results
for image_path, decoded, elapsed in results:
    print(f"Image: {image_path}")
    print("Model output:", decoded)
    print(f"Inference time: {elapsed:.2f} seconds\n")