import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

def main(args):
    """Main training function."""
    
    # Load model and tokenizer
    model_name = "mistralai/Mistral-7B-v0.1"
    print(f"--- [DEBUG] Loading base model: {model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("--- [DEBUG] Model and tokenizer loaded. ---")

    # dataset load 
    training_data_path = os.path.join(args.training_dir, "dataset_filtered.jsonl")
    print(f"--- [DEBUG] Loading dataset from: {training_data_path} ---")

    try:
        raw_dataset = load_dataset('json', data_files=training_data_path, split='train')
        print(f"--- [DEBUG] Successfully loaded raw dataset with {len(raw_dataset)} rows. ---")
    except Exception as e:
        print(f"--- [FATAL ERROR] Failed to load dataset. Error: {e} ---")
        return # Exit if we can't even load the file

    # This function will now print what it finds
    def preprocess_and_tokenize(examples):
        texts_to_tokenize = []
        for i in range(len(examples['content'])):
            text = examples['content'][i]
            
            if not text:
                embeds = examples['embeds'][i]
                if embeds and isinstance(embeds, list) and len(embeds) > 0:
                    description = embeds[0].get('description')
                    if description:
                        text = description
            
            
            if i < 20: # Only print the first 20 for brevity
                # Truncate the text to 150 characters for clean logs
                log_text = (text[:150] + '...') if text and len(text) > 150 else text
                print(f"--- [DEBUG] Row {i} | Extracted Text: '{log_text}' ---")

            texts_to_tokenize.append(text if text else "")
        
        return tokenizer(texts_to_tokenize, truncation=True, max_length=128)

    tokenized_dataset = raw_dataset.map(
        preprocess_and_tokenize, 
        batched=True, 
        remove_columns=raw_dataset.column_names
    )
    
    print(f"--- [DEBUG] Tokenized dataset now has {len(tokenized_dataset)} rows. ---")
    
    # Exit gracefully if the dataset is empty after processing
    if len(tokenized_dataset) == 0:
        print("--- [FATAL ERROR] Dataset is empty after preprocessing. No data to train on. Exiting. ---")
    
        dummy_file_path = os.path.join(args.model_dir, "training_failed.log")
        with open(dummy_file_path, "w") as f:
            f.write("Dataset was empty. See CloudWatch logs for details.")
        print("--- [DEBUG] Dummy file created. ---")
        return

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    print("--- [DEBUG] Dataset prepared and tokenized. ---")

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_strategy="no",
        logging_steps=100,
        fp16=False,
        bf16=True,
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    
    # Start training
    print("--- [DEBUG] Starting model training... ---")
    trainer.train()
    print("--- [DEBUG] Training complete. ---")
    
    # Save final model 
    print(f"--- [DEBUG] Saving final model artifacts to {args.model_dir} ---")
    trainer.save_model(args.model_dir)
    print("--- [DEBUG] Model saving complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--training_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    args = parser.parse_args()
    main(args)
