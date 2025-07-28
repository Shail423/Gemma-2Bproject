from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from datasets import Dataset

import torch

# 1. DATA: Replace with your actual prompt/response/image-description
data = [
    {
        "prompt": "Describe this image of a cat.",
        "response": "A fluffy white cat is sitting on a wooden table looking at the camera."
    }
]

# 2. DATASET PREP
dataset = Dataset.from_list(data)

# 3. TOKENIZER/MODEL
model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# 4. TOKENIZATION
def preprocess(example):
    text = f"<start_of_turn>user\n{example['prompt']}<end_of_turn>\n<start_of_turn>model\n{example['response']}<end_of_turn>\n"
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(preprocess)

# 5. TRAIN/VAL SPLIT (only if >1 example)
if len(tokenized_dataset) > 1:
    train_test = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test["train"]
    eval_dataset = train_test["test"]
else:
    train_dataset = tokenized_dataset
    eval_dataset = None

# 6. TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir="./gemma-finetuned-single",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    fp16=torch.cuda.is_available(),      # Use FP16 if on GPU
    save_total_limit=1,
    logging_steps=1,
    report_to="none",
    save_strategy="no",
   # evaluation_strategy="no"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 7. TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# 8. TRAIN
trainer.train()
print("✅ Training complete! Checkpoints in ./gemma-finetuned-single")
