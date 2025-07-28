from transformers import TrainingArguments  # type: ignore
from transformers import Trainer  # type: ignore
from transformers import DataCollatorForLanguageModeling  # type: ignore

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

dataset = load_dataset("json", data_files={"train": "train.jsonl"})

def preprocess(batch):
    prompts = batch["prompt"]
    responses = batch["response"]
    texts = [
        f"<start_of_turn>user\n{p}<end_of_turn>\n<start_of_turn>model\n{r}<end_of_turn>\n"
        for p, r in zip(prompts, responses)
    ]
    encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=256)
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

tokenized = dataset.map(preprocess, batched=True)
train_data = tokenized["train"]

training_args = TrainingArguments(
    output_dir="./gemma-finetuned-single",
    per_device_train_batch_size=1,
    num_train_epochs=4,
    fp16=True,
    logging_steps=1,
    save_total_limit=1,
    report_to="none",
    push_to_hub=False,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    data_collator=data_collator,
)

trainer.train()
