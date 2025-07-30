from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset, DatasetDict  # Explicit import for types
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU usage

def check_fields(filepath):
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if isinstance(data, dict):
                    return list(data.keys())
    return []

def preprocess_function(batch, tokenizer, prompt_field, response_field):
    texts = [
        f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"
        for prompt, response in zip(batch[prompt_field], batch[response_field])
    ]
    # FIX: Use encode_plus for batch processing with __call__ method
    encodings = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = encodings['input_ids'].tolist()
    attention_mask = encodings['attention_mask'].tolist()
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': input_ids
    }

def main():
    model_id = "google/gemma-2b-it"  # Safer for CPU environments
    train_file = "D:/Gemma-2Bproject/invoices.jsonl"
    valid_file = "D:/Gemma-2Bproject/invoices_valid.jsonl"

    fields = check_fields(train_file)
    if not fields or len(fields) < 2:
        raise ValueError("Could not determine fields in the training JSONL file")

    if 'prompt' in fields and 'response' in fields:
        prompt_field, response_field = 'prompt', 'response'
    elif 'input' in fields and 'output' in fields:
        prompt_field, response_field = 'input', 'output'
    elif 'input_text' in fields and 'target_text' in fields:
        prompt_field, response_field = 'input_text', 'target_text'
    else:
        prompt_field, response_field = fields[0], fields[1]
        print(f"Warning: Using first two fields as prompt and response: {prompt_field}, {response_field}")

    print(f"Using fields -> prompt: {prompt_field}, response: {response_field}")

    # FIX: No declared type, just get it as is and inspect type at runtime
    dataset = load_dataset("json", data_files={"train": train_file, "validation": valid_file})

    train_dataset = dataset["train"]
    valid_dataset = dataset["validation"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # Make sure to only ever pass Datasets, not lists
    remove_cols = train_dataset.column_names if hasattr(train_dataset, "column_names") else list(train_dataset[0].keys())

    def preprocess(batch):
        return preprocess_function(batch, tokenizer, prompt_field, response_field)

    tokenized_datasets = dataset.map(
        preprocess,
        batched=True,
        remove_columns=remove_cols
    )

    tokenized_train = tokenized_datasets["train"] if isinstance(tokenized_datasets["train"], Dataset) else Dataset.from_dict(tokenized_datasets["train"])
    tokenized_valid = tokenized_datasets["validation"] if isinstance(tokenized_datasets["validation"], Dataset) else Dataset.from_dict(tokenized_datasets["validation"])

    batch_size = 1
    gradient_accumulation_steps = 4

    training_args = TrainingArguments(
        output_dir="./models",
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        weight_decay=0.01,
        learning_rate=1e-4,
        fp16=False,
        no_cuda=True,  # CPU only!
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()

if __name__ == "__main__":
    main()
