from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import torch

accelerator = Accelerator()
device = torch.device("cpu")


model_id = "google/gemma-3n-e2b-it"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  # This will automatically determine the best device for each part of the model (CPU/GPU)
    offload_folder="D:/Gemma-2Bproject/offload",  # Specify the offload folder for disk storage
)

# Prepare model with Accelerator to handle large model
model = accelerator.prepare(model)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Model is now ready for inference or further tasks
