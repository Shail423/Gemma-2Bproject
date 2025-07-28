import torch
import time
import concurrent.futures
from utils import load_model_and_processor

# --- Device selection: Use GPU if available, else CPU ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Load model, processor, and move model to correct device ---
model, processor, _ = load_model_and_processor()
model.to(device)
print("Model loaded on device:", device)

# --- Disable gradients for inference ---
torch.set_grad_enabled(False)

# --- List of image paths to process ---
image_paths = [
    "input_images/invoice13.jpg",
    # Add more image paths if needed
]

def run_inference(image_path):
    try:
        # You can update this prompt logic as needed
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
        # Prepare input for the model, ensure moved to correct device
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
                use_cache=True
            )
        # If your processor uses input_ids, ensure slicing is correct
        # Some processors may have a different interface, so adjust if needed
        new_tokens = generation[0][inputs["input_ids"].shape[-1]:]
        decoded = processor.decode(new_tokens, skip_special_tokens=True)
        elapsed = time.time() - start_time
        return image_path, decoded, elapsed
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return image_path, str(e), 0

# --- Run inference in parallel for all images ---
with concurrent.futures.ThreadPoolExecutor(max_workers=2 if device.type=='cuda' else 4) as executor:
    results = list(executor.map(run_inference, image_paths))

# --- Print results ---
for image_path, decoded, elapsed in results:
    print(f"\nImage: {image_path}")
    print("Model output:", decoded)
    print(f"Inference time: {elapsed:.2f} seconds\n")
