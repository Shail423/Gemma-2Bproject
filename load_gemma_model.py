from transformers import AutoProcessor, Gemma3nForConditionalGeneration

model_id = "google/gemma-3n-e2b-it"

# Load the model
model = Gemma3nForConditionalGeneration.from_pretrained(model_id)
print("Model loaded successfully!")

# Load the processor (tokenizer, etc.)
processor = AutoProcessor.from_pretrained(model_id)
print("Processor (tokenizer) loaded successfully!") 