from transformers import AutoProcessor, AutoModelForSeq2SeqLM  # Use AutoModelForSeq2SeqLM
import torch

def load_model_and_processor():
    model_id = "google/gemma-3n-e2b-it"
    
    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    # Load the processor
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Load device (e.g., CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, processor, device
