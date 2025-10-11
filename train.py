import os
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling,
)

def main(args):
    """Main training function."""
    
    # 1. --- Load Model and Tokenizer ---
    # This is where you specify the exact model you want to fine-tune.
    model_name = "mistralai/Mistral-7B-v0.1"
    
    print(f"Loading base model: {model_name}")
    
    # Load the pre-trained model and tokenizer from Hugging Face.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically use the available GPU.
        torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency.
    )
    
    # The Mistral model doesn't have a default padding token.
    # We'll set it to the End-Of-Sentence token for training.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Model and tokenizer loaded successfully.")

    # 2. --- Load and Prepare the Dataset ---
    # SageMaker copies your S3 data to the path specified by the 'training' channel.
    training_data_file = "dataset_filtered.jsonl"
    training_data_path = os.path.join(args.training_dir, training_data_file)
    
    print(f"Loading and preparing dataset from: {training_data_path}")

    # The TextDataset class is a simple way to load a text file for language modeling.
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=training_data_path,
        block_size=128,  # This chunks the text into blocks of 128 tokens. Adjust if needed.
    )
    
    # This collator prepares batches of data for language modeling.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,  # Causal language modeling, not masked language modeling.
    )
    
    print("Dataset prepared.")

    # 3. --- Define Training Arguments ---
    # These arguments control the training process.
    training_args = TrainingArguments(
        output_dir=args.model_dir,  # **Crucial:** SageMaker requires the output to be here.
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_strategy="no",  # Don't save checkpoints; only save the final model.
        logging_steps=100,
        fp16=False,  # We are using bfloat16 instead.
        bf16=True,   # Enable bfloat16 for better performance on modern GPUs.
    )
    
    # 4. --- Initialize the Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    # 5. --- Start Training ---
    print("Starting model training...")
    trainer.train()
    print("Training complete.")
    
    # 6. --- Save the Final Model ---
    # The Trainer saves the model, tokenizer, and config to the specified output directory.
    print(f"Saving final model artifacts to {args.model_dir}")
    trainer.save_model(args.model_dir)
    print("Model saving complete.")


if __name__ == "__main__":
    # This block parses arguments passed by the SageMaker Estimator in your notebook.
    parser = argparse.ArgumentParser()
    
    # Hyperparameters from the Estimator
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    
    # SageMaker's environment variables for file paths
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--training_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    
    args = parser.parse_args()
    main(args)
