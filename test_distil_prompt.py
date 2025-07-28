from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# ======= CONFIGURE YOUR MODEL =======
# model_id = "distilgpt2"
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # <-- Use chat model for Q&A!
# model_id = "Qwen/Qwen1.5-0.5B-Chat"

prompt = "Who is the current Prime Minister of India?"

# ======= DEVICE SELECTION =======
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU for inference.")
else:
    device = torch.device("cpu")
    print("Warning: GPU unavailable or busy, using CPU.")

# ======= LOAD MODEL AND TOKENIZER =======
print(f"Loading model {model_id}...")
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# ======= SMART INPUT HANDLING =======
def prepare_inputs(tokenizer, prompt, device):
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
    if isinstance(inputs, dict):
        input_ids = inputs["input_ids"].to(device)
    else:
        input_ids = inputs.to(device)
    return input_ids

input_ids = prepare_inputs(tokenizer, prompt, device)
input_len = input_ids.shape[-1]

# ======= INFERENCE WITH TIMING =======
start_time = time.time()
with torch.no_grad():
    outputs = model.generate(input_ids, max_new_tokens=64)
    generated = outputs[0][input_len:]
inference_time = time.time() - start_time
result = tokenizer.decode(generated, skip_special_tokens=True)

# ======= PRINT OUTPUT =======
print("\nPrompt:", prompt)
print("Model output:\n" + result)
print(f"\nInference time: {inference_time:.2f} seconds")

if result.strip() == "":
    print("\n[NOTE] Try using a chat-tuned model like TinyLlama/TinyLlama-1.1B-Chat-v1.0 for better Q&A.")
