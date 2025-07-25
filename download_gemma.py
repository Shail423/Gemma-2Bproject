from transformers import AutoProcessor, AutoModelForSeq2SeqLM

# Model ID for Gemma
model_id = "google/gemma-3n-e2b-it"

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
print("Model loaded successfully!")

# Load the processor (tokenizer, etc.)
processor = AutoProcessor.from_pretrained(model_id)
print("Processor (tokenizer) loaded successfully!")
