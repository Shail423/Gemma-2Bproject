from transformers import AutoTokenizer, AutoModelForCausalLM

# Model ID for Gemma
model_id = "google/gemma-3n-e2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
