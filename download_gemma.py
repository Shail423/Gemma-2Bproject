from transformers import AutoTokenizer, AutoModelForCausalLM

# Model ID for Gemma
model_id = "google/distil-gemma-2b" 

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
