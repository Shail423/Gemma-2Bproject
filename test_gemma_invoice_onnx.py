import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import onnx
import onnxruntime as ort
import numpy as np
import os

# Set up model and tokenizer
model_id = "google/gemma-3n-e2b-it"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Example prompt
prompt = "Extract invoice_number, date, vendor, total_amount, and line_items as JSON from this invoice: Widget A, 2x25.00; Widget B, 1x75.50."
inputs = tokenizer(prompt, return_tensors="pt")

onnx_model_path = "gemma-3n-e2b-it.onnx"

# Export to ONNX (try-catch for learning)
try:
    print("Exporting model to ONNX format...")
    torch.onnx.export(
        model,
        (inputs["input_ids"],),
        onnx_model_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence"}, "logits": {0: "batch_size", 1: "sequence"}},
        opset_version=16,
        do_constant_folding=True,
    )
    print(f"Model exported to {onnx_model_path}")
except Exception as e:
    print(f"ONNX export failed: {e}")
    exit()

# Load ONNX model and run inference (if export succeeded)
try:
    print("Loading ONNX model and running inference with ONNX Runtime...")
    ort_session = ort.InferenceSession(onnx_model_path)
    ort_inputs = {"input_ids": inputs["input_ids"].cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    logits = ort_outs[0]
    # Get the most likely next token
    next_token_id = np.argmax(logits[0, -1, :])
    next_token = tokenizer.decode([next_token_id])
    print(f"Next token predicted by ONNX model: {next_token}")
except Exception as e:
    print(f"ONNX Runtime inference failed: {e}") 