from transformers import AutoProcessor, Gemma3nForConditionalGeneration

model_id = "google/gemma-3n-E2B-it"

# Download and cache the model weights
model = Gemma3nForConditionalGeneration.from_pretrained(model_id)
# Download and cache the processor
processor = AutoProcessor.from_pretrained(model_id)

print("Model and processor downloaded successfully!")