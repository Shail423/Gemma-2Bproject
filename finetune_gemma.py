from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import torch
import os
import json
from typing import Union, Optional

def check_data_format(file_path):
    """Check the format of JSONL files and return field names"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line:
                data = json.loads(first_line)
                return list(data.keys())
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return []

def gemma_fine_tuning_complete():
    print("=== GEMMA FINE-TUNING - COMPLETE FIXED VERSION ===")
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Available GPU memory: {gpu_memory:.1f} GB")
        
        
        if gpu_memory >= 16:
            train_batch_size = 4
            eval_batch_size = 8
            gradient_accumulation = 4
        elif gpu_memory >= 12:
            train_batch_size = 2
            eval_batch_size = 4
            gradient_accumulation = 8
        else:
            train_batch_size = 1
            eval_batch_size = 2
            gradient_accumulation = 16
    else:
        train_batch_size = 1
        eval_batch_size = 1
        gradient_accumulation = 4
    
    print(f"Optimized batch configuration:")
    print(f"  - Train batch size: {train_batch_size}")
    print(f"  - Eval batch size: {eval_batch_size}")
    print(f"  - Gradient accumulation: {gradient_accumulation}")
    print(f"  - Effective batch size: {train_batch_size * gradient_accumulation}")

    
    model_id = "google/gemma-2b-it"  
    print(f"Using model: {model_id}")

    print("\n=== LOADING TOKENIZER ===")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded successfully!")
    except Exception as e:
        print(f"❌ Tokenizer loading failed: {e}")
        return

    print("\n=== LOADING MODEL (FIXED FOR GTX 1650) ===")
    try:
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        
        if torch.cuda.is_available():
            model = model.to('cuda')
        
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Direct loading failed: {e}")
        print("Trying 4-bit quantization...")
        
        try:
            
            from transformers.utils.quantization_config import BitsAndBytesConfig

            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                trust_remote_code=True
            )
            print("✅ Model loaded with 4-bit quantization!")
            
        except Exception as e2:
            print(f"❌ 4-bit loading failed: {e2}")
            print("Falling back to CPU...")
            
            try:
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                print("✅ Model loaded on CPU!")
                
            except Exception as e3:
                print(f"❌ All loading methods failed: {e3}")
                return

    print("\n=== CHECKING DATA FORMAT ===")
    train_file = "D:/Gemma-2Bproject/invoices.jsonl"
    val_file = "D:/Gemma-2Bproject/invoices_valid.jsonl"
    
   
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print(f"❌ Data files not found!")
        print(f"Train file exists: {os.path.exists(train_file)}")
        print(f"Validation file exists: {os.path.exists(val_file)}")
        return
    
    
    train_fields = check_data_format(train_file)
    val_fields = check_data_format(val_file)
    
    print(f"Train file fields: {train_fields}")
    print(f"Validation file fields: {val_fields}")
    
    
    prompt_field = None
    response_field = None
    
    
    if "prompt" in train_fields and "response" in train_fields:
        prompt_field, response_field = "prompt", "response"
    elif "input" in train_fields and "output" in train_fields:
        prompt_field, response_field = "input", "output"
    elif "text" in train_fields and "label" in train_fields:
        prompt_field, response_field = "text", "label"
    elif "question" in train_fields and "answer" in train_fields:
        prompt_field, response_field = "question", "answer"
    elif "input_text" in train_fields and "target_text" in train_fields:
        prompt_field, response_field = "input_text", "target_text"
    elif len(train_fields) >= 2:
        
        prompt_field, response_field = train_fields[0], train_fields[1]
        print(f"⚠️ Using first two fields as prompt/response: {prompt_field}, {response_field}")
    else:
        print(f"❌ Cannot determine field names from: {train_fields}")
        return
    
    print(f"✅ Using fields - Prompt: '{prompt_field}', Response: '{response_field}'")

    print("\n=== LOADING DATASETS ===")
    try:
        dataset = load_dataset(
            "json",
            data_files={
                "train": train_file,
                "validation": val_file
            }
        )
      
        
        raw_train_ds = dataset["train"]
        raw_val_ds = dataset["validation"]
        
       
        if not isinstance(raw_train_ds, Dataset) or not isinstance(raw_val_ds, Dataset):
            raise TypeError("Loaded datasets are not proper Dataset objects")
        
        train_ds: Dataset = raw_train_ds
        val_ds: Dataset = raw_val_ds
        
        print(f"✅ Datasets loaded - Train: {len(train_ds)}, Validation: {len(val_ds)}")
        
        if len(train_ds) == 0 or len(val_ds) == 0:
            print("❌ One or both datasets are empty!")
            return
            
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return

    print("\n=== PREPROCESSING WITH DYNAMIC FIELD NAMES ===")
    def preprocess_batch(batch):
        """Preprocessing function with dynamic field names"""
        try:
            texts = [
                f"<start_of_turn>user\n{p}<end_of_turn>\n<start_of_turn>model\n{r}<end_of_turn>\n"
                for p, r in zip(batch[prompt_field], batch[response_field])
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
        except KeyError as e:
            print(f"❌ Field not found: {e}")
            print(f"Available fields: {list(batch.keys())}")
            raise
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            raise

    try:
        print("Processing training dataset...")
        tokenized_datasets = dataset.map(
            preprocess_batch, 
            batched=True,
            batch_size=1000,
            remove_columns=train_ds.column_names
        )
        
        
        raw_tokenized_train = tokenized_datasets["train"]
        raw_tokenized_val = tokenized_datasets["validation"]
        
        
        if not isinstance(raw_tokenized_train, Dataset) or not isinstance(raw_tokenized_val, Dataset):
            raise TypeError("Tokenized datasets are not proper Dataset objects")
        
        tokenized_train: Dataset = raw_tokenized_train
        tokenized_val: Dataset = raw_tokenized_val
        
        print(f"✅ Preprocessing completed!")
        print(f"  - Tokenized train: {len(tokenized_train)}")
        print(f"  - Tokenized validation: {len(tokenized_val)}")
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n=== SETTING UP TRAINING ===")
    
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )
    
    
    training_args = TrainingArguments(
        output_dir="./gemma-finetuned-fixed",
        
        
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        
        
        num_train_epochs=3,
        eval_strategy="steps",
        eval_steps=10,  
        save_strategy="steps",
        save_steps=20,
        
        
        learning_rate=2e-5,
        warmup_steps=10,  
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        
        fp16=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        
        
        logging_steps=5,
        logging_dir="./logs",
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        
        remove_unused_columns=False,
    )

    print("Training configuration:")
    print(f"  - Effective batch size: {train_batch_size * gradient_accumulation}")
    print(f"  - Evaluation every: {training_args.eval_steps} steps")
    print(f"  - Total epochs: {training_args.num_train_epochs}")

    print("\n=== CREATING TRAINER ===")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    print("\n=== STARTING TRAINING ===")
    try:
        
        train_result = trainer.train()
        
        print("🎉 Training completed successfully!")
        print(f"Final training loss: {train_result.training_loss:.4f}")
        
        
        final_model_path = "./gemma-finetuned-final"
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print(f"✅ Model saved to: {final_model_path}")
        
        
        print("\n=== FINAL EVALUATION ===")
        eval_results = trainer.evaluate()
        print(f"Final evaluation loss: {eval_results['eval_loss']:.4f}")
        
        return trainer, final_model_path
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def create_sample_data():
    """Create sample data files if they don't exist"""
    train_file = "D:/Gemma-2Bproject/invoices.jsonl"
    val_file = "D:/Gemma-2Bproject/invoices_valid.jsonl"
    
    if not os.path.exists(train_file) or not os.path.exists(val_file):
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
        
        
        with open(train_file, "w", encoding="utf-8") as f:
            for item in sample_data * 12:  
                f.write(json.dumps(item) + "\n")
        
        
        with open(val_file, "w", encoding="utf-8") as f:
            for item in sample_data:  
                f.write(json.dumps(item) + "\n")
        
        print(f"✅ Sample data created: {train_file}, {val_file}")

if __name__ == "__main__":
   
    result = gemma_fine_tuning_complete()
    
    if result and result[0] and result[1]:
        trainer, model_path = result
        print(f"\n🚀 Fine-tuning completed successfully!")
        print(f"📁 Model saved at: {model_path}")
        print("🎯 You can now use your fine-tuned Gemma model for invoice extraction!")
    else:
        print("\n❌ Fine-tuning failed. Please check the error messages above.")       