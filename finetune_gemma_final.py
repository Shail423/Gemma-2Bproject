from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import torch
import os
import json
from typing import List, Dict, Any, Union


os.environ["CUDA_VISIBLE_DEVICES"] = ""


def detect_field_names(filepath: str) -> List[str]:
    """Detect field names from JSONL file by reading first non-empty line"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        return list(data.keys())
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return []


def safe_get_columns(dataset) -> List[str]:
    """Safely get column names from dataset object"""
    if hasattr(dataset, 'column_names') and dataset.column_names is not None:
        return list(dataset.column_names)
    elif hasattr(dataset, 'features') and dataset.features is not None:
        return list(dataset.features.keys())
    elif isinstance(dataset, list) and len(dataset) > 0 and isinstance(dataset[0], dict):
        return list(dataset[0].keys())
    return []


def ensure_dataset(obj: Union[Dataset, list, Any]) -> Dataset:
    """Ensure the object is a proper Dataset instance"""
    if isinstance(obj, Dataset):
        return obj
    elif isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
        
        keys = list(obj[0].keys())
        data_dict = {key: [item[key] for item in obj] for key in keys}
        return Dataset.from_dict(data_dict)
    else:
        
        try:
            return Dataset.from_dict({"data": [str(obj)]})
        except:
            raise TypeError(f"Cannot convert {type(obj)} to Dataset")


def preprocess_batch(batch: Dict[str, List[Any]], tokenizer, prompt_field: str, response_field: str) -> Dict[str, List[Any]]:
    """Preprocess batch with proper formatting for Gemma model"""
    texts = [
        f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"
        for prompt, response in zip(batch[prompt_field], batch[response_field])
    ]
    
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors=None
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main():
    print("=== GEMMA FINE-TUNING - FINAL FIXED VERSION ===")
    print("Using CPU-only training for GTX 1650 compatibility")
    
    
    model_id = "google/gemma-2b-it"  
    train_file = "D:/Gemma-2Bproject/invoices.jsonl"
    valid_file = "D:/Gemma-2Bproject/invoices_valid.jsonl"
    
    
    if not os.path.exists(train_file) or not os.path.exists(valid_file):
        print(f"❌ Data files not found!")
        print(f"Train file: {train_file} - Exists: {os.path.exists(train_file)}")
        print(f"Valid file: {valid_file} - Exists: {os.path.exists(valid_file)}")
        return
    
    
    print("\n=== DETECTING DATA FIELDS ===")
    fields = detect_field_names(train_file)
    if not fields or len(fields) < 2:
        print(f"❌ Could not detect fields in {train_file}")
        return
    
    
    if "prompt" in fields and "response" in fields:
        prompt_field, response_field = "prompt", "response"
    elif "input" in fields and "output" in fields:
        prompt_field, response_field = "input", "output"
    elif "input_text" in fields and "target_text" in fields:
        prompt_field, response_field = "input_text", "target_text"
    elif "text" in fields and "label" in fields:
        prompt_field, response_field = "text", "label"
    else:
        prompt_field, response_field = fields[0], fields[1]
        print(f"⚠️ Using first two fields: {prompt_field}, {response_field}")
    
    print(f"✅ Detected fields - Prompt: '{prompt_field}', Response: '{response_field}'")

    
    print("\n=== LOADING TOKENIZER AND MODEL ===")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded successfully!")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  
            low_cpu_mem_usage=True
        )
        print("✅ Model loaded successfully on CPU!")
        
    except Exception as e:
        print(f"❌ Error loading model/tokenizer: {e}")
        return
    
    
    print("\n=== LOADING DATASETS ===")
    try:
        dataset = load_dataset(
            "json",
            data_files={"train": train_file, "validation": valid_file}
        )
        
        
        train_dataset = ensure_dataset(dataset["train"])
        valid_dataset = ensure_dataset(dataset["validation"])
        
        
        train_size = len(train_dataset)
        valid_size = len(valid_dataset)
        
        print(f"✅ Datasets loaded - Train: {train_size}, Valid: {valid_size}")
        
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return
    
    
    print("\n=== PREPROCESSING DATASETS ===")
    def preprocess(batch):
        return preprocess_batch(batch, tokenizer, prompt_field, response_field)
    
    
    columns_to_remove = safe_get_columns(train_dataset)
    
    try:
        tokenized_datasets = dataset.map(
            preprocess,
            batched=True,
            remove_columns=columns_to_remove
        )
        
        
        tokenized_train = ensure_dataset(tokenized_datasets["train"])
        tokenized_valid = ensure_dataset(tokenized_datasets["validation"])
        
        print(f"✅ Preprocessing completed!")
        print(f"  - Tokenized train: {len(tokenized_train)}")
        print(f"  - Tokenized valid: {len(tokenized_valid)}")
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return
    
    
    print("\n=== SETTING UP TRAINING ===")
    training_args = TrainingArguments(
        output_dir="./gemma-finetuned-cpu",
        
        
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  
        
        
        num_train_epochs=3,
        eval_strategy="epoch",  
        save_strategy="epoch",
        
        
        learning_rate=2e-5,
        warmup_steps=10,
        weight_decay=0.01,
        
        
        use_cpu=True,  
        fp16=False,    
        dataloader_num_workers=2,
        
        
        logging_steps=5,
        logging_dir="./logs",
        save_total_limit=2,
        report_to="none",
        
        
        dataloader_pin_memory=False,
        remove_unused_columns=True,
    )
    
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  
    )
    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,  
        eval_dataset=tokenized_valid,   
        data_collator=data_collator,
    )
    
    print("Training configuration:")
    print(f"  - Device: CPU")
    print(f"  - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  - Total epochs: {training_args.num_train_epochs}")
    print(f"  - Learning rate: {training_args.learning_rate}")
    
    
    print("\n=== STARTING TRAINING ===")
    print("⏰ Note: CPU training will be slower but stable!")
    
    try:
        train_result = trainer.train()
        
        print("🎉 Training completed successfully!")
        print(f"Final training loss: {train_result.training_loss:.4f}")
        
        
        final_model_path = "./gemma-cpu-finetuned-final"
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print(f"✅ Model saved to: {final_model_path}")
        
        
        print("\n=== FINAL EVALUATION ===")
        eval_results = trainer.evaluate()
        print(f"Final evaluation loss: {eval_results['eval_loss']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_sample_data():
    """Create sample data if files don't exist"""
    train_file = "D:/Gemma-2Bproject/invoices.jsonl"
    valid_file = "D:/Gemma-2Bproject/invoices_valid.jsonl"
    
    if not os.path.exists(train_file) or not os.path.exists(valid_file):
        print("Creating sample data files...")
        
        sample_data = [
            {
                "input_text": "Invoice No: 12345\nDate: 2024-01-15\nCompany: ABC Corp\nAmount: $1000.00",
                "target_text": '{"invoice_number": "12345", "date": "2024-01-15", "company": "ABC Corp", "amount": "$1000.00"}'
            },
            {
                "input_text": "Invoice No: 67890\nDate: 2024-02-20\nCompany: XYZ Ltd\nAmount: $500.50", 
                "target_text": '{"invoice_number": "67890", "date": "2024-02-20", "company": "XYZ Ltd", "amount": "$500.50"}'
            }
        ]
        
        
        os.makedirs(os.path.dirname(train_file), exist_ok=True)
        with open(train_file, "w", encoding="utf-8") as f:
            for item in sample_data * 12:  # 
                f.write(json.dumps(item) + "\n")
        
        
        with open(valid_file, "w", encoding="utf-8") as f:
            for item in sample_data:  
                f.write(json.dumps(item) + "\n")
        
        print(f"✅ Sample data created: {train_file}, {valid_file}")


if __name__ == "__main__":
    print("=== GEMMA FINE-TUNING - ALL PYLANCE ERRORS FIXED ===")
    print("Fixed all type assignment errors:")
    print("  ✅ Lines 126-127: Dataset assignment errors fixed")
    print("  ✅ Lines 156-157: Tokenized dataset assignment errors fixed")
    print("  ✅ Added ensure_dataset() function for type safety")
    print("  ✅ Guaranteed Dataset objects passed to Trainer")
    print()
    
    
    
    success = main()
    
    if success:
        print("\n🚀 Fine-tuning completed successfully!")
        print("📁 Your trained model is ready for invoice extraction!")
    else:
        print("\n❌ Fine-tuning failed. Check error messages above.")
