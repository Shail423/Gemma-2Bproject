from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model_and_processor():
    model_id = "google/gemma-3n-e2b-it"
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # Load the tokenizer
    processor = AutoTokenizer.from_pretrained(model_id)
    
    # Load device (e.g., CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, processor, device
