from transformers import AutoProcessor, Gemma3nForConditionalGeneration

model_id = "google/gemma-3n-e2b-it"

model = Gemma3nForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

print("Model and processor downloaded successfully!") 