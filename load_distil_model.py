from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Path to your locally-downloaded model directory
model_id = "./distilgpt2"

print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
print("Model and tokenizer loaded successfully!")

# Select device: GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded on device: {device}")

# Example prompt (you can change this to your invoice test prompt)
prompt = "Extract invoice_number, date, vendor, total_amount from this invoice: Vendor: ABC, Invoice No: 123, Date: 2024-07-28, Total: $99.99"

print("\nRunning inference...")
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=64)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nPrompt:")
print(prompt)
print("\nModel output:")
print(decoded)
