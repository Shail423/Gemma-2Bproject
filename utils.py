import torch
from transformers import AutoProcessor, AutoModelForConditionalGeneration

def load_model_and_processor(model_id="google/gemma-3n-e2b-it"):
    """Loads the Gemma model and processor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AutoModelForConditionalGeneration.from_pretrained(model_id).to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor, device