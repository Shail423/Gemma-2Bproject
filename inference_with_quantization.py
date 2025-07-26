import torch
import time
import concurrent.futures
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Define the model identifier or local path
model_id = "google/gemma-3n-E2B-it"  # or "D:/Gemma-2Bproject/models/gemma_model" for local path

# Step 2: Authenticate and load the model and tokenizer (use token for private models)
token = "your_huggingface_token_here"  # Replace with your actual Hugging Face token (for private models)

# Load model with dynamic quantization and half-precision for faster inference
model = AutoModelForCausalLM.from_pretrained(model_id, token=token, torch_dtype="auto")  # Auto dtype for efficient memory
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

# Apply dynamic quantization (helps with memory usage and faster inference)
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
print("Dynamic quantization applied to the model.")

# Step 3: Move model to GPU if available (otherwise, CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
model.to(device)
print(f"Model loaded on device: {device}")

# Step 4: Disable gradients for faster inference
torch.set_grad_enabled(False)

# List of image paths to process in parallel (you can modify these paths based on your actual images)
image_paths = [
    "input_images/invoice13.jpg",  # Add more image paths if needed
    # You can add more image paths as needed
]

# Step 5: Define the function to run inference for each image
def run_inference(image_path):
    try:
        # Generate a description of the image (this is where you'd customize based on your use case)
        image_description = f"This is an invoice image located at {image_path}. Please extract invoice_number, date, vendor, total_amount, and line_items (description, quantity, unit_price, line_total) as JSON."

        # Prepare the input for the model (tokenize the description)
        inputs = tokenizer(image_description, return_tensors="pt").to(device)  # Ensure inputs are on CPU or GPU

        start_time = time.time()  # Start time for inference

        # Run inference without gradient calculation for faster performance
        with torch.no_grad():
            generation = model.generate(
                **inputs,
                max_new_tokens=256,  # Control the length of the output
                do_sample=False,  # We don’t want randomness in the generation
                use_cache=True  # Use cached computations if possible
            )

        # Decode the output from the model
        decoded = tokenizer.decode(generation[0], skip_special_tokens=True)
        elapsed = time.time() - start_time  # Measure inference time

        return image_path, decoded, elapsed
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return image_path, str(e), 0

# Step 6: Run inference in parallel for all images
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_inference, image_paths))

# Step 7: Print results
for image_path, decoded, elapsed in results:
    print(f"Image: {image_path}")
    print("Model output:", decoded)
    print(f"Inference time: {elapsed:.2f} seconds\n")
