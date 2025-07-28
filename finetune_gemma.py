from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset

model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Example: Load a text dataset (replace with your path)
dataset = load_dataset("json", data_files={"train": "train.jsonl", "validation": "val.jsonl"})

def preprocess(batch):
    # For prompt-response, join as single text (instruction tuning)
    prompt = batch["prompt"]
    response = batch["response"]
    text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>\n"
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(preprocess, batched=False)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./gemma-finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    fp16=True,  # Use fp16 if you have a supported GPU
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
)

trainer.train()
