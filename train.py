import os
import argparse
import torch
from datasets import load_dataset # <-- IMPORT THE CORRECT LIBRARY
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

def main(args):
    """Main training function."""
    
    # 1. --- Load Model and Tokenizer ---
    model_name = "mistralai/Mistral-7B-v0.1"
    
    print(f"Loading base model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Model and tokenizer loaded successfully.")

    # 2. --- Load and Prepare the Dataset (CORRECTED) ---
    training_data_file = "dataset_filtered.jsonl"
    training_data_path = os.path.join(args.training_dir, training_data_file)
    
    print(f"Loading dataset from: {training_data_path}")

    # Use load_dataset for .jsonl files
    raw_dataset = load_dataset('json', data_files=training_data_path, split='train')

    # IMPORTANT: Assumes the text is in a column named "content". 
    # If your JSONL key is different (e.g., "text"), change it here.
    def tokenize_function(examples):
        return tokenizer(examples["content"], truncation=True, max_length=128)

    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=raw_dataset.column_names)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
    )
    
    print("Dataset prepared and tokenized.")

    # 3. --- Define Training Arguments ---
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_strategy="no",
        logging_steps=100,
        fp16=False,
        bf16=True,
    )
    
    # 4. --- Initialize the Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset, # <-- USE THE NEW TOKENIZED DATASET
    )
    
    # 5. --- Start Training ---
    print("Starting model training...")
    trainer.train()
    print("Training complete.")
    
    # 6. --- Save the Final Model ---
    print(f"Saving final model artifacts to {args.model_dir}")
    trainer.save_model(args.model_dir)
    print("Model saving complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--training_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    
    args = parser.parse_args()
    main(args)
