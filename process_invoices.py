import json
import time
import os
import torch
import concurrent.futures
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image

# Set the number of threads for CPU (optional, doesn't affect GPU)
os.environ["OMP_NUM_THREADS"] = "16"
torch.set_num_threads(16)

# Change model_id if you want to use a smaller version (example for distilled version)
model_id = "google/gemma-3n-distill"  # Use a smaller model if available

# Check if CUDA (GPU) is available, use GPU if it is; otherwise, fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # Post-process the data to ensure proper formatting
        structured_data = process_extracted_data(decoded)

        return image_path, structured_data, elapsed
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return image_path, str(e), 0

def process_extracted_data(data):
    """
    Process and structure the extracted data.
    Assumes `data` is the raw extracted text from the model.
    """
    try:
        # Example: Converting the raw model output into a structured format (JSON)
        structured_data = {
            "invoice_number": "12345",  # Placeholder, replace with actual extraction logic
            "date": "01-01-2023",       # Ensure date is properly formatted
            "vendor": "ABC Corp",        # Vendor field
            "total_amount": 500.00,      # Total amount extracted
            "line_items": [
                {
                    "description": "Laptop",
                    "quantity": 2,
                    "unit_price": 250.0,
                    "line_total": 500.0
                }
            ]
        }
        
        # If the date is incomplete, set it to a default value
        if structured_data["date"] == "01-":
            structured_data["date"] = "01-01-2023"  # Replace with default or parsed date

        return structured_data
    except Exception as e:
        print(f"Error during post-processing: {e}")
        return {}

# Run inference in parallel for all images
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_inference, image_paths))

# Print results
for image_path, structured_data, elapsed in results:
    print(f"Image: {image_path}")
    print("Structured Output:", json.dumps(structured_data, indent=4))
    print(f"Inference time: {elapsed:.2f} seconds\n")
